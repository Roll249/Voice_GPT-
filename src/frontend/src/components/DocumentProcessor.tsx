import React, { useRef, useState } from 'react';
import axios from 'axios';

interface DocumentProcessorProps {
  modelType: 'chatterbox' | 'xtts';
  exaggeration: number;
  isLoading: boolean;
  onAudioGenerated: (url: string) => void;
  referenceUploaded: boolean;
}

export const DocumentProcessor: React.FC<DocumentProcessorProps> = ({
  modelType,
  exaggeration,
  isLoading,
  onAudioGenerated,
  referenceUploaded
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!referenceUploaded) {
      setError('Please upload a reference audio first');
      return;
    }

    if (!file.type.includes('pdf')) {
      setError('Please upload a PDF file');
      return;
    }

    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('exaggeration', exaggeration.toString());
      formData.append('model_type', modelType);

      const response = await axios.post('/api/process-pdf', formData, {
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
      console.error('Error processing document:', err);
      setError('Failed to process document. Please try again.');
    } finally {
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <section className="document-processor">
      <h2>3. Document to Speech</h2>
      
      <div className="doc-upload-area">
        <button
          className="doc-upload-btn"
          onClick={() => fileInputRef.current?.click()}
          disabled={isLoading || !referenceUploaded}
        >
          {isLoading ? 'Processing...' : 'Select PDF Document'}
        </button>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileUpload}
          accept=".pdf"
          style={{ display: 'none' }}
        />
        <p>Upload a PDF document to convert to speech</p>
      </div>
      
      {error && <div className="error-message">{error}</div>}
    </section>
  );
};