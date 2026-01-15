import { useState, useRef } from 'react'
import axios from 'axios'
import './App.css'
import { VoiceSelector } from './components/VoiceSelector'
import { TextToSpeech } from './components/TextToSpeech'
import { DocumentProcessor } from './components/DocumentProcessor'

function App() {
  const [modelType, setModelType] = useState<'chatterbox' | 'xtts'>('chatterbox')
  const [exaggeration, setExaggeration] = useState<number>(0.5)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [referenceUploaded, setReferenceUploaded] = useState<boolean>(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleReferenceUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    // Check file type
    if (!file.type.includes('audio')) {
      setUploadError('Please upload an audio file (WAV, MP3, etc.)')
      return
    }

    setUploadError(null)
    setIsLoading(true)

    try {
      const formData = new FormData()
      formData.append('file', file)

      await axios.post('/api/upload-reference', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setReferenceUploaded(true)
      alert('Reference audio uploaded successfully!')
    } catch (err) {
      console.error('Error uploading reference:', err)
      setUploadError('Failed to upload reference audio. Please try again.')
      setReferenceUploaded(false)
    } finally {
      setIsLoading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const handleAudioGenerated = (url: string) => {
    // Revoke previous URL to prevent memory leaks
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl)
    }
    setAudioUrl(url)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎙️ Vietnamese Text-to-Speech</h1>
        <p>High-quality voice cloning for Vietnamese language</p>
      </header>

      <main className="app-main">
        <section className="reference-upload">
          <h2>1. Upload Reference Voice</h2>
          <div className="upload-area">
            <button
              className="upload-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
            >
              {isLoading ? 'Uploading...' : referenceUploaded ? '✓ Reference Uploaded - Upload New' : 'Upload Audio File'}
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleReferenceUpload}
              accept="audio/*"
              style={{ display: 'none' }}
            />
            <p>Upload a clear audio sample (WAV, MP3, etc.) of the voice you want to clone</p>
            {uploadError && <div className="error-message">{uploadError}</div>}
          </div>
        </section>

        <VoiceSelector selectedModel={modelType} onModelChange={setModelType} />

        <TextToSpeech
          modelType={modelType}
          exaggeration={exaggeration}
          setExaggeration={setExaggeration}
          isLoading={isLoading}
          onAudioGenerated={handleAudioGenerated}
          referenceUploaded={referenceUploaded}
        />

        <DocumentProcessor
          modelType={modelType}
          exaggeration={exaggeration}
          isLoading={isLoading}
          onAudioGenerated={handleAudioGenerated}
          referenceUploaded={referenceUploaded}
        />

        {audioUrl && (
          <section className="audio-player">
            <h2>Generated Audio</h2>
            <audio controls src={audioUrl} />
            <a href={audioUrl} download="generated_speech.wav" className="download-btn">
              Download Audio
            </a>
          </section>
        )}
      </main>

      <footer className="app-footer">
        <p>Powered by Chatterbox-TTS and XTTS-v2</p>
      </footer>
    </div>
  )
}

export default App
