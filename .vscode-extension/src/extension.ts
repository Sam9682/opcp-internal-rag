import * as vscode from 'vscode';
import { DeploymentsTreeProvider } from './deploymentsTreeProvider';

export function activate(context: vscode.ExtensionContext) {
    const provider = new DeploymentsTreeProvider();

    const treeView = vscode.window.createTreeView('deploymentsView', {
        treeDataProvider: provider,
        showCollapseAll: true,
    });

    context.subscriptions.push(
        treeView,
        vscode.commands.registerCommand('deployments.refresh', () => provider.refresh()),
        vscode.commands.registerCommand('deployments.openComposeFile', (uri: vscode.Uri) => {
            vscode.window.showTextDocument(uri);
        }),
        vscode.commands.registerCommand('deployments.openInBrowser', (port: string) => {
            vscode.env.openExternal(vscode.Uri.parse(`http://localhost:${port}`));
        }),
    );

    // Auto-refresh when compose files change
    const watcher = vscode.workspace.createFileSystemWatcher('**/docker-compose*.{yml,yaml}');
    watcher.onDidChange(() => provider.refresh());
    watcher.onDidCreate(() => provider.refresh());
    watcher.onDidDelete(() => provider.refresh());
    context.subscriptions.push(watcher);
}

export function deactivate() {}
