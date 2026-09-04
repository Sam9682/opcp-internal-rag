import React from 'react';
import { Source } from '../types';

interface SourceCardProps {
  source: Source;
}

const SourceCard: React.FC<SourceCardProps> = ({ source }) => {
  return (
    <div className="bg-gray-50 border border-gray-200 rounded-md p-2 sm:p-3 text-xs sm:text-sm">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="font-semibold text-gray-800 mb-1 truncate">{source.title}</div>
          <div className="text-gray-600 text-xs mb-2 line-clamp-2">{source.excerpt}</div>
          {source.file_path && (
            <div className="text-xs text-gray-500 font-mono truncate">{source.file_path}</div>
          )}
        </div>
        <div className="flex-shrink-0">
          <div className="text-xs font-semibold text-blue-600">
            {(source.similarity_score * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-gray-500">match</div>
        </div>
      </div>
    </div>
  );
};

export default SourceCard;
