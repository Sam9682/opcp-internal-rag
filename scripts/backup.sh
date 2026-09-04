#!/bin/bash

# Backup script for OPCP Internal RAG application
# This script creates backups of the database and important configuration files

set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="./backups"
DATABASE_BACKUP_FILE="$BACKUP_DIR/database_backup_$TIMESTAMP.sql"
CONFIG_BACKUP_DIR="$BACKUP_DIR/configs_$TIMESTAMP"

# Create backup directory
mkdir -p "$BACKUP_DIR"
mkdir -p "$CONFIG_BACKUP_DIR"

echo "Starting backup of OPCP Internal RAG application..."

# Backup database
echo "Backing up database..."
if docker-compose exec postgres pg_dump -U rag_user rag_db > "$DATABASE_BACKUP_FILE"; then
    echo "✓ Database backup completed: $DATABASE_BACKUP_FILE"
else
    echo "✗ Failed to backup database"
    exit 1
fi

# Backup configuration files
echo "Backing up configuration files..."
cp .env* "$CONFIG_BACKUP_DIR/" 2>/dev/null || true
cp -r conf/ "$CONFIG_BACKUP_DIR/" 2>/dev/null || true
cp -r certs/ "$CONFIG_BACKUP_DIR/" 2>/dev/null || true

echo "✓ Configuration backup completed: $CONFIG_BACKUP_DIR"

echo "Backup completed successfully!"
echo "Files created:"
echo "  - $DATABASE_BACKUP_FILE"
echo "  - $CONFIG_BACKUP_DIR/"