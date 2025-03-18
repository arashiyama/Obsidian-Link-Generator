import { App, PluginSettingTab, Setting } from 'obsidian';
import AutoLinkPlugin from './main';

export class AutoLinkSettingTab extends PluginSettingTab {
    plugin: AutoLinkPlugin;

    constructor(app: App, plugin: AutoLinkPlugin) {
        super(app, plugin);
        this.plugin = plugin;
    }

    display(): void {
        const { containerEl } = this;
        containerEl.empty();

        containerEl.createEl('h2', { text: 'Auto Link Settings' });

        new Setting(containerEl)
            .setName('OpenAI API Key')
            .setDesc('Your OpenAI API key for generating embeddings')
            .addText(text => text
                .setPlaceholder('Enter your OpenAI API key')
                .setValue(this.plugin.settings.openAIApiKey)
                .onChange(async (value) => {
                    this.plugin.settings.openAIApiKey = value;
                    await this.plugin.saveSettings();
                })
            );

        new Setting(containerEl)
            .setName('API Endpoint')
            .setDesc('API endpoint for embeddings (default is OpenAI)')
            .addText(text => text
                .setPlaceholder('https://api.openai.com/v1/embeddings')
                .setValue(this.plugin.settings.apiEndpoint)
                .onChange(async (value) => {
                    this.plugin.settings.apiEndpoint = value;
                    await this.plugin.saveSettings();
                })
            );

        new Setting(containerEl)
            .setName('Embedding Model')
            .setDesc('The OpenAI model to use for embeddings')
            .addText(text => text
                .setPlaceholder('text-embedding-ada-002')
                .setValue(this.plugin.settings.embeddingModel)
                .onChange(async (value) => {
                    this.plugin.settings.embeddingModel = value;
                    await this.plugin.saveSettings();
                })
            );

        new Setting(containerEl)
            .setName('Semantic Threshold')
            .setDesc('Minimum similarity score (0-1) for suggesting links')
            .addSlider(slider => slider
                .setLimits(0, 1, 0.05)
                .setValue(this.plugin.settings.semanticThreshold)
                .setDynamicTooltip()
                .onChange(async (value) => {
                    this.plugin.settings.semanticThreshold = value;
                    await this.plugin.saveSettings();
                })
            );

        new Setting(containerEl)
            .setName('Maximum Links')
            .setDesc('Maximum number of links to add per note')
            .addSlider(slider => slider
                .setLimits(1, 50, 1)
                .setValue(this.plugin.settings.maxLinksPerNote)
                .setDynamicTooltip()
                .onChange(async (value) => {
                    this.plugin.settings.maxLinksPerNote = value;
                    await this.plugin.saveSettings();
                })
            );

        new Setting(containerEl)
            .setName('Include Tags')
            .setDesc('Consider note tags when calculating similarity')
            .addToggle(toggle => toggle
                .setValue(this.plugin.settings.includeTags)
                .onChange(async (value) => {
                    this.plugin.settings.includeTags = value;
                    await this.plugin.saveSettings();
                })
            );

        new Setting(containerEl)
            .setName('Include Metadata')
            .setDesc('Consider note metadata (frontmatter) when calculating similarity')
            .addToggle(toggle => toggle
                .setValue(this.plugin.settings.includeMetadata)
                .onChange(async (value) => {
                    this.plugin.settings.includeMetadata = value;
                    await this.plugin.saveSettings();
                })
            );

        new Setting(containerEl)
            .setName('Link Section Heading')
            .setDesc('Heading for the section where links are added')
            .addText(text => text
                .setPlaceholder('## Related Notes')
                .setValue(this.plugin.settings.linkSection)
                .onChange(async (value) => {
                    this.plugin.settings.linkSection = value;
                    await this.plugin.saveSettings();
                })
            );

        new Setting(containerEl)
            .setName('Processing Batch Size')
            .setDesc('Number of notes to process in each batch (higher = faster but more memory usage)')
            .addSlider(slider => slider
                .setLimits(10, 200, 10)
                .setValue(this.plugin.settings.processingBatchSize)
                .setDynamicTooltip()
                .onChange(async (value) => {
                    this.plugin.settings.processingBatchSize = value;
                    await this.plugin.saveSettings();
                })
            );

        new Setting(containerEl)
            .setName('Show Progress Notices')
            .setDesc('Show notifications during batch processing')
            .addToggle(toggle => toggle
                .setValue(this.plugin.settings.showProgressNotices)
                .onChange(async (value) => {
                    this.plugin.settings.showProgressNotices = value;
                    await this.plugin.saveSettings();
                })
            );

        new Setting(containerEl)
            .setName('Debug Mode')
            .setDesc('Enable additional logging for troubleshooting')
            .addToggle(toggle => toggle
                .setValue(this.plugin.settings.debugMode)
                .onChange(async (value) => {
                    this.plugin.settings.debugMode = value;
                    await this.plugin.saveSettings();
                })
            );
    }
}
