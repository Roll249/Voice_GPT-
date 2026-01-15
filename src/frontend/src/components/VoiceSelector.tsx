import React from 'react';

interface VoiceSelectorProps {
  selectedModel: 'chatterbox' | 'xtts';
  onModelChange: (model: 'chatterbox' | 'xtts') => void;
}

export const VoiceSelector: React.FC<VoiceSelectorProps> = ({ 
  selectedModel, 
  onModelChange 
}) => {
  return (
    <div className="model-selector">
      <h3>Select Voice Model</h3>
      
      <div className="model-option">
        <input
          type="radio"
          id="chatterbox"
          name="model"
          checked={selectedModel === 'chatterbox'}
          onChange={() => onModelChange('chatterbox')}
        />
        <label htmlFor="chatterbox">
          <strong>Chatterbox</strong> - High quality, MIT license, excellent for Vietnamese
        </label>
      </div>
      
      <div className="model-option">
        <input
          type="radio"
          id="xtts"
          name="model"
          checked={selectedModel === 'xtts'}
          onChange={() => onModelChange('xtts')}
        />
        <label htmlFor="xtts">
          <strong>XTTS-v2</strong> - Multilingual, good cross-language support
        </label>
      </div>
    </div>
  );
};