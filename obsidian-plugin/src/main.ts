import { App, Editor, MarkdownView, Modal, Notice, Plugin, PluginSettingTab, Setting } from 'obsidian';
import { AutoLinkSettings, DEFAULT_SETTINGS } from './settings';
import { AutoLinkSettingTab } from './settingsTab';
import { AutoLinkModel } from './model';

export default class AutoLinkPlugin extends Plugin {
	settings: AutoLinkSettings;
	model: AutoLinkModel;

	async onload() {
		await this.loadSettings();
		
		// Initialize the model
		this.model = new AutoLinkModel(this);

		// Add the settings tab
		this.addSettingTab(new AutoLinkSettingTab(this.app, this));

		// Register commands
		this.addCommand({
			id: 'generate-semantic-links',
			name: 'Generate Semantic Links for Current Note',
			editorCallback: async (editor: Editor, view: MarkdownView) => {
				const file = view.file;
				if (!file) {
					new Notice('No file is currently open.');
					return;
				}
				
				new Notice('Generating semantic links...');
				try {
					const success = await this.model.generateLinksForFile(file);
					if (success) {
						new Notice('Semantic links generated successfully!');
					} else {
						new Notice('Failed to generate semantic links.');
					}
				} catch (error) {
					console.error('Error generating semantic links:', error);
					new Notice('Error generating semantic links: ' + error.message);
				}
			}
		});

		this.addCommand({
			id: 'generate-all-semantic-links',
			name: 'Generate Semantic Links for All Notes',
			callback: async () => {
				new Notice('Generating semantic links for all notes...');
				try {
					const count = await this.model.generateLinksForAllFiles();
					new Notice(`Semantic links generated for ${count} notes!`);
				} catch (error) {
					console.error('Error generating semantic links for all notes:', error);
					new Notice('Error generating semantic links: ' + error.message);
				}
			}
		});

		// Register status bar
		const statusBarItem = this.addStatusBarItem();
		statusBarItem.setText('Auto-Link: Ready');

		// When the plugin is loaded, log a message
		console.log('Auto Link plugin loaded');
	}

	onunload() {
		console.log('Auto Link plugin unloaded');
	}

	async loadSettings() {
		this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
	}

	async saveSettings() {
		await this.saveData(this.settings);
	}
}
