import { TFile, Vault, Notice } from 'obsidian';
import AutoLinkPlugin from './main';

interface NoteContent {
    title: string;
    content: string;
    tags: string[];
    metadata: Record<string, any>;
    path: string;
}

interface EmbeddingResponse {
    data: {
        embedding: number[];
    }[];
    usage: {
        prompt_tokens: number;
        total_tokens: number;
    };
}

interface NoteWithEmbedding extends NoteContent {
    embedding: number[];
}

interface SimilarNote {
    path: string;
    title: string;
    similarity: number;
}

export class AutoLinkModel {
    private plugin: AutoLinkPlugin;
    private embeddings: Map<string, number[]> = new Map();
    private processing = false;

    constructor(plugin: AutoLinkPlugin) {
        this.plugin = plugin;
    }

    async generateLinksForFile(file: TFile): Promise<boolean> {
        if (this.processing) {
            new Notice('Already processing. Please wait...');
            return false;
        }

        this.processing = true;
        try {
            const allNotes = await this.getAllNotes();
            const noteContents = await this.extractNoteContents(file);
            
            if (!noteContents) {
                new Notice('Failed to extract note contents');
                return false;
            }

            const noteEmbedding = await this.getEmbedding(this.prepareTextForEmbedding(noteContents));
            if (!noteEmbedding) {
                new Notice('Failed to generate embedding');
                return false;
            }

            // Store this note's embedding
            this.embeddings.set(file.path, noteEmbedding);

            // Find similar notes
            const similarNotes = await this.findSimilarNotes(
                { ...noteContents, embedding: noteEmbedding },
                allNotes
            );

            // Update the note with links
            return await this.updateNoteWithLinks(file, similarNotes);
        } catch (error) {
            console.error('Error generating links:', error);
            new Notice(`Error: ${error.message}`);
            return false;
        } finally {
            this.processing = false;
        }
    }

    async generateLinksForAllFiles(): Promise<number> {
        if (this.processing) {
            new Notice('Already processing. Please wait...');
            return 0;
        }

        this.processing = true;
        let processedCount = 0;

        try {
            const vault = this.plugin.app.vault;
            const markdownFiles = vault.getMarkdownFiles();
            const batchSize = this.plugin.settings.processingBatchSize;
            const showNotices = this.plugin.settings.showProgressNotices;

            // Process in batches
            for (let i = 0; i < markdownFiles.length; i += batchSize) {
                const batch = markdownFiles.slice(i, i + batchSize);
                if (showNotices) {
                    new Notice(`Processing batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(markdownFiles.length / batchSize)}`);
                }

                // Process each file in the batch
                for (const file of batch) {
                    const success = await this.generateLinksForFile(file);
                    if (success) processedCount++;
                }
            }

            return processedCount;
        } catch (error) {
            console.error('Error processing all notes:', error);
            new Notice(`Error: ${error.message}`);
            return processedCount;
        } finally {
            this.processing = false;
        }
    }

    private async getAllNotes(): Promise<NoteWithEmbedding[]> {
        const vault = this.plugin.app.vault;
        const markdownFiles = vault.getMarkdownFiles();
        const notes: NoteWithEmbedding[] = [];

        for (const file of markdownFiles) {
            const noteContents = await this.extractNoteContents(file);
            if (!noteContents) continue;

            let embedding: number[] = [];
            
            // Check if we already have an embedding for this file
            if (this.embeddings.has(file.path)) {
                embedding = this.embeddings.get(file.path)!;
            } else {
                // Generate a new embedding
                embedding = await this.getEmbedding(this.prepareTextForEmbedding(noteContents)) || [];
                if (embedding.length > 0) {
                    this.embeddings.set(file.path, embedding);
                }
            }

            if (embedding.length > 0) {
                notes.push({
                    ...noteContents,
                    embedding
                });
            }
        }

        return notes;
    }

    private async extractNoteContents(file: TFile): Promise<NoteContent | null> {
        try {
            const content = await this.plugin.app.vault.read(file);
            const title = file.basename;

            // Extract tags (simple regex for now)
            const tagRegex = /#[a-zA-Z0-9_-]+/g;
            const tags = content.match(tagRegex) || [];

            // Extract frontmatter metadata (simplified)
            const metadata: Record<string, any> = {};
            const frontmatterRegex = /^---\n([\s\S]*?)\n---/;
            const frontmatterMatch = content.match(frontmatterRegex);
            
            if (frontmatterMatch && frontmatterMatch[1]) {
                const frontmatter = frontmatterMatch[1];
                const lines = frontmatter.split('\n');
                
                for (const line of lines) {
                    const parts = line.split(':');
                    if (parts.length >= 2) {
                        const key = parts[0].trim();
                        const value = parts.slice(1).join(':').trim();
                        metadata[key] = value;
                    }
                }
            }

            return {
                title,
                content,
                tags,
                metadata,
                path: file.path
            };
        } catch (error) {
            console.error(`Error extracting contents from ${file.path}:`, error);
            return null;
        }
    }

    private prepareTextForEmbedding(note: NoteContent): string {
        let text = `Title: ${note.title}\n\n${note.content}`;
        
        // Add tags if configured
        if (this.plugin.settings.includeTags && note.tags.length > 0) {
            text += `\n\nTags: ${note.tags.join(' ')}`;
        }
        
        // Add metadata if configured
        if (this.plugin.settings.includeMetadata && Object.keys(note.metadata).length > 0) {
            text += '\n\nMetadata:';
            for (const [key, value] of Object.entries(note.metadata)) {
                text += `\n${key}: ${value}`;
            }
        }
        
        return text;
    }

    private async getEmbedding(text: string): Promise<number[] | null> {
        if (!this.plugin.settings.openAIApiKey) {
            new Notice('API key not configured in settings');
            return null;
        }

        try {
            const response = await fetch(this.plugin.settings.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.plugin.settings.openAIApiKey}`
                },
                body: JSON.stringify({
                    input: text,
                    model: this.plugin.settings.embeddingModel
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(`API error: ${error.error?.message || response.statusText}`);
            }

            const data = await response.json() as EmbeddingResponse;
            return data.data[0].embedding;
        } catch (error) {
            console.error('Error generating embedding:', error);
            return null;
        }
    }

    private async findSimilarNotes(
        currentNote: NoteWithEmbedding,
        allNotes: NoteWithEmbedding[]
    ): Promise<SimilarNote[]> {
        const similarNotes: SimilarNote[] = [];
        const threshold = this.plugin.settings.semanticThreshold;
        const maxLinks = this.plugin.settings.maxLinksPerNote;

        for (const note of allNotes) {
            // Skip the current note
            if (note.path === currentNote.path) continue;

            // Calculate cosine similarity
            const similarity = this.calculateCosineSimilarity(currentNote.embedding, note.embedding);

            if (similarity >= threshold) {
                similarNotes.push({
                    path: note.path,
                    title: note.title,
                    similarity
                });
            }
        }

        // Sort by similarity (descending) and take top matches
        return similarNotes
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, maxLinks);
    }

    private calculateCosineSimilarity(vec1: number[], vec2: number[]): number {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors must have the same dimensions');
        }

        let dotProduct = 0;
        let mag1 = 0;
        let mag2 = 0;

        for (let i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            mag1 += vec1[i] * vec1[i];
            mag2 += vec2[i] * vec2[i];
        }

        mag1 = Math.sqrt(mag1);
        mag2 = Math.sqrt(mag2);

        if (mag1 === 0 || mag2 === 0) return 0;
        return dotProduct / (mag1 * mag2);
    }

    private async updateNoteWithLinks(file: TFile, similarNotes: SimilarNote[]): Promise<boolean> {
        try {
            const content = await this.plugin.app.vault.read(file);
            const sectionHeading = this.plugin.settings.linkSection;
            let updatedContent = content;

            // Find existing section or create a new one
            const sectionRegex = new RegExp(`${this.escapeRegExp(sectionHeading)}\\s*(?:\\n(?!#)[^\\n]*)*`, 'g');
            const sectionMatch = sectionRegex.exec(content);

            // Format links
            const linksText = similarNotes.map(note => {
                const percentage = Math.round(note.similarity * 100);
                return `- [[${note.title}]] (${percentage}% similarity)`;
            }).join('\n');

            if (sectionMatch) {
                // Update existing section
                updatedContent = content.replace(
                    sectionMatch[0],
                    `${sectionHeading}\n${linksText}`
                );
            } else {
                // Add new section at the end of the note
                updatedContent = content + `\n\n${sectionHeading}\n${linksText}`;
            }

            // Only update if content has changed
            if (updatedContent !== content) {
                await this.plugin.app.vault.modify(file, updatedContent);
            }

            return true;
        } catch (error) {
            console.error(`Error updating note ${file.path}:`, error);
            return false;
        }
    }

    private escapeRegExp(string: string): string {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
}
