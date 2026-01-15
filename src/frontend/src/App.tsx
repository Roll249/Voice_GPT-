import React, { useState, useRef } from 'react';
import './App.css';
import axios from 'axios';
import { VoiceSelector } from './components/VoiceSelector';
import { TextToSpeech } from './components/TextToSpeech';
import { DocumentProcessor } from './components/DocumentProcessor';

function App() {
  const [selectedModel, setSelectedModel] = useState<'chatterbox' | 'xtts'>('chatterbox');
  const [exaggeration, setExaggeration] = useState<number>(0.5);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [referenceUploaded, setReferenceUploaded] = useState<boolean>(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleReferenceUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('/api/upload-reference', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.status === 'success') {
        setReferenceUploaded(true);
        alert('Reference audio uploaded successfully!');
      }
    } catch (error) {
      console.error('Error uploading reference:', error);
      alert('Failed to upload reference audio');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePlayAudio = () => {
    if (audioUrl) {
      const audio = new Audio(audioUrl);
      audio.play();
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>Vietnamese Text-to-Speech Application</h1>
        <p>Clone voices and generate Vietnamese speech from text or documents</p>
      </header>

      <main className="app-main">
        {/* Voice Selection */}
        <section className="voice-section">
          <h2>1. Upload Reference Voice</h2>
          <div className="upload-area">
            <button 
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              className="upload-btn"
            >
              {isLoading ? 'Uploading...' : 'Select Audio Reference'}
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleReferenceUpload}
              accept="audio/*"
              style={{ display: 'none' }}
            />
            {referenceUploaded && (
              <span className="status-success">✓ Reference uploaded</span>
            )}
          </div>
          
          <VoiceSelector 
            selectedModel={selectedModel} 
            onModelChange={setSelectedModel}
          />
        </section>

        {/* Text to Speech */}
        <TextToSpeech 
          modelType={selectedModel}
          exaggeration={exaggeration}
          setExaggeration={setExaggeration}
          isLoading={isLoading}
          onAudioGenerated={setAudioUrl}
          referenceUploaded={referenceUploaded}
        />

        {/* Document Processor */}
        <DocumentProcessor 
          modelType={selectedModel}
          exaggeration={exaggeration}
          isLoading={isLoading}
          onAudioGenerated={setAudioUrl}
          referenceUploaded={referenceUploaded}
        />
      </main>

      {audioUrl && (
        <div className="audio-controls">
          <button onClick={handlePlayAudio}>Play Audio</button>
          <a href={audioUrl} download="generated-speech.wav">Download Audio</a>
        </div>
      )}
    </div>
  );
}

export default App;