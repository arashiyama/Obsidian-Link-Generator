export interface AutoLinkSettings {
    openAIApiKey: string;
    apiEndpoint: string;
    embeddingModel: string;
    semanticThreshold: number;
    maxLinksPerNote: number;
    includeTags: boolean;
    includeMetadata: boolean;
    linkSection: string;
    processingBatchSize: number;
    showProgressNotices: boolean;
    debugMode: boolean;
}

export const DEFAULT_SETTINGS: AutoLinkSettings = {
    openAIApiKey: '',
    apiEndpoint: 'https://api.openai.com/v1/embeddings',
    embeddingModel: 'text-embedding-ada-002',
    semanticThreshold: 0.75,
    maxLinksPerNote: 10,
    includeTags: true,
    includeMetadata: true,
    linkSection: '## Related Notes',
    processingBatchSize: 50,
    showProgressNotices: true,
    debugMode: false
}
