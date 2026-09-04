import * as vscode from 'vscode';
import * as path from 'path';
import { parseComposeFile, ComposeService } from './composeParser';

export type TreeItem = ComposeFileItem | ServiceItem | PortItem;

export class DeploymentsTreeProvider implements vscode.TreeDataProvider<TreeItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<TreeItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    refresh(): void {
        this._onDidChangeTreeData.fire(undefined);
    }

    getTreeItem(element: TreeItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: TreeItem): Promise<TreeItem[]> {
        if (!element) {
            return this.getComposeFiles();
        }
        if (element instanceof ComposeFileItem) {
            return this.getServices(element);
        }
        if (element instanceof ServiceItem) {
            return this.getPorts(element);
        }
        return [];
    }

    private async getComposeFiles(): Promise<ComposeFileItem[]> {
        const files = await vscode.workspace.findFiles(
            '**/docker-compose*.{yml,yaml}',
            '**/node_modules/**'
        );
        if (files.length === 0) {
            return [];
        }
        return files
            .sort((a, b) => a.fsPath.localeCompare(b.fsPath))
            .map(uri => new ComposeFileItem(uri));
    }

    private async getServices(file: ComposeFileItem): Promise<ServiceItem[]> {
        const services = await parseComposeFile(file.resourceUri);
        return services.map(s => new ServiceItem(s, file.resourceUri));
    }

    private getPorts(service: ServiceItem): PortItem[] {
        return service.service.ports.map(p => new PortItem(p, service.service.name));
    }
}
