import React, { useState } from 'react';
import axios from 'axios';

interface TextToSpeechProps {
  modelType: 'chatterbox' | 'xtts';
  exaggeration: number;
  setExaggeration: (value: number) => void;
  isLoading: boolean;
  onAudioGenerated: (url: string) => void;
  referenceUploaded: boolean;
}

export const TextToSpeech: React.FC<TextToSpeechProps> = ({
  modelType,
  exaggeration,
  setExaggeration,
  isLoading,
  onAudioGenerated,
  referenceUploaded
}) => {
  const [text, setText] = useState<string>('Xin chào, đây là một bài kiểm tra giọng nói tiếng Việt.');
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!referenceUploaded) {
      setError('Please upload a reference audio first');
      return;
    }

    if (!text.trim()) {
      setError('Please enter some text to generate speech');
      return;
    }

    setError(null);
    try {
      const formData = new FormData();
      formData.append('text', text);
      formData.append('exaggeration', exaggeration.toString());
      formData.append('model_type', modelType);

      const response = await axios.post('/api/generate', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob',
      });

      // Create a blob URL for the audio
      const audioBlob = new Blob([response.data], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(audioBlob);
      onAudioGenerated(audioUrl);
    } catch (err) {
      console.error('Error generating speech:', err);
      setError('Failed to generate speech. Please try again.');
    }
  };

  return (
    <section className="text-to-speech">
      <h2>2. Text to Speech</h2>
      
      <textarea
        className="text-input"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter Vietnamese text here..."
        disabled={isLoading || !referenceUploaded}
      />
      
      <div className="exaggeration-control">
        <label>
          Emotion Level: {exaggeration.toFixed(2)}
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={exaggeration}
          onChange={(e) => setExaggeration(parseFloat(e.target.value))}
          disabled={isLoading || !referenceUploaded}
        />
      </div>
      
      <button
        className="generate-btn"
        onClick={handleGenerate}
        disabled={isLoading || !referenceUploaded}
      >
        {isLoading ? 'Generating...' : 'Generate Speech'}
      </button>
      
      {error && <div className="error-message">{error}</div>}
    </section>
  );
};