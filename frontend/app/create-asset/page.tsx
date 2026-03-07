'use client';

import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import dynamic from 'next/dynamic';
import Link from 'next/link';

const GLBViewer = dynamic(() => import('@/components/GLBViewer'), { ssr: false });

interface Message {
  role: 'user' | 'assistant';
  content: string;
  image?: string;
  imageUrl?: string;
}

type Step = 1 | 2;
type Step1State = 'idle' | 'generating';
type Step2State = 'idle' | 'generating' | 'complete' | 'saving';

export default function CreateAssetPage() {
  const router = useRouter();

  // Step management
  const [currentStep, setCurrentStep] = useState<Step>(1);

  // Step 1: Gemini chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [step1State, setStep1State] = useState<Step1State>('idle');
  const [currentImage, setCurrentImage] = useState<string | null>(null);
  const [currentImageUrl, setCurrentImageUrl] = useState<string | null>(null);

  // Step 2: Meshy state
  const [step2State, setStep2State] = useState<Step2State>('idle');
  const [progress, setProgress] = useState(0);
  const [glbUrl, setGlbUrl] = useState<string | null>(null);
  const [sourceImageKey, setSourceImageKey] = useState<string | null>(null);
  const [modelKey, setModelKey] = useState<string | null>(null);

  // Shared
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Step 1: Send message to Gemini
  const handleSendMessage = async () => {
    if (!input.trim() || step1State === 'generating') return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setStep1State('generating');
    setError(null);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...messages, userMessage],
          currentImage,
        }),
      });

      const data = await response.json();

      if (data.error) {
        setError(data.error);
        setStep1State('idle');
        return;
      }

      const assistantMessage: Message = {
        role: 'assistant',
        content: data.text || 'Here\'s your image!',
        image: data.image,
        imageUrl: data.imageUrl,
      };

      setMessages((prev) => [...prev, assistantMessage]);

      if (data.image) {
        setCurrentImage(data.image);
        setCurrentImageUrl(data.imageUrl);
      }

      setStep1State('idle');
    } catch (err) {
      console.error('Error:', err);
      setError('Failed to generate image');
      setStep1State('idle');
    }
  };

  // Move to Step 2
  const handleProceedToStep2 = () => {
    if (!currentImage) return;
    setCurrentStep(2);
    setStep2State('idle');
    setGlbUrl(null);
    setSourceImageKey(null);
    setModelKey(null);
    setError(null);
  };

  // Go back to Step 1
  const handleBackToStep1 = () => {
    setCurrentStep(1);
    setStep2State('idle');
    setGlbUrl(null);
  };

  // Step 2: Generate 3D with Meshy
  const handleGenerate3D = async () => {
    if (!currentImage) return;

    setStep2State('generating');
    setProgress(0);
    setError(null);

    try {
      const startResponse = await fetch('/api/meshy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageBase64: currentImage }),
      });

      const startData = await startResponse.json();

      if (startData.error) {
        setError(startData.error);
        setStep2State('idle');
        return;
      }

      const taskId = startData.taskId;
      setSourceImageKey(startData.sourceImageKey);

      // Poll for completion
      const pollInterval = setInterval(async () => {
        try {
          const statusResponse = await fetch(`/api/meshy?taskId=${taskId}`);
          const statusData = await statusResponse.json();

          if (statusData.error) {
            clearInterval(pollInterval);
            setError(statusData.error);
            setStep2State('idle');
            return;
          }

          setProgress(statusData.progress || 0);

          if (statusData.status === 'complete') {
            clearInterval(pollInterval);
            setGlbUrl(statusData.glbUrl);
            setModelKey(statusData.modelKey);
            setStep2State('complete');
          } else if (statusData.status === 'failed') {
            clearInterval(pollInterval);
            setError(statusData.error || 'Failed to generate 3D model');
            setStep2State('idle');
          }
        } catch (err) {
          console.error('Poll error:', err);
        }
      }, 3000);

    } catch (err) {
      console.error('Error:', err);
      setError('Failed to start 3D generation');
      setStep2State('idle');
    }
  };

  // Regenerate 3D
  const handleRegenerate = () => {
    setGlbUrl(null);
    setModelKey(null);
    handleGenerate3D();
  };

  // Save to library and go to library
  const handleApproveAndSave = async () => {
    if (!sourceImageKey || !modelKey) {
      // Just download if we don't have keys
      if (glbUrl) {
        window.open(glbUrl, '_blank');
      }
      return;
    }

    setStep2State('saving');

    try {
      const response = await fetch('/api/assets/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: `Asset ${new Date().toLocaleDateString()}`,
          sourceImageKey,
          modelKey,
        }),
      });

      const data = await response.json();

      if (data.error) {
        setError(data.error);
        setStep2State('complete');
        return;
      }

      // Redirect to library
      router.push('/library');
    } catch (err) {
      console.error('Error saving:', err);
      setError('Failed to save asset');
      setStep2State('complete');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Step Indicator */}
      <div className="border-b border-gray-700 p-4">
        <div className="max-w-4xl mx-auto flex items-center gap-4">
          <Link href="/library" className="text-gray-400 hover:text-white mr-4">
            ← Library
          </Link>
          <div className={`flex items-center gap-2 ${currentStep === 1 ? 'text-blue-400' : 'text-gray-500'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${currentStep === 1 ? 'bg-blue-600' : 'bg-gray-700'}`}>
              1
            </div>
            <span className="font-medium">Generate Image</span>
          </div>
          <div className="flex-1 h-px bg-gray-700" />
          <div className={`flex items-center gap-2 ${currentStep === 2 ? 'text-green-400' : 'text-gray-500'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${currentStep === 2 ? 'bg-green-600' : 'bg-gray-700'}`}>
              2
            </div>
            <span className="font-medium">Create 3D Asset</span>
          </div>
        </div>
      </div>

      {/* Step 1: Gemini Chat */}
      {currentStep === 1 && (
        <div className="flex h-[calc(100vh-73px)]">
          {/* Chat Panel */}
          <div className="w-1/2 flex flex-col border-r border-gray-700">
            <div className="p-4 border-b border-gray-700">
              <h2 className="text-lg font-semibold">Step 1: Generate Image</h2>
              <p className="text-gray-400 text-sm">Chat with AI to create and refine your image</p>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.length === 0 && (
                <div className="text-gray-500 text-center py-8">
                  <p>Describe what you want to create.</p>
                  <p className="text-sm mt-2">Example: &quot;A red ceramic vase with floral patterns&quot;</p>
                </div>
              )}

              {messages.map((msg, i) => (
                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] rounded-lg p-3 ${msg.role === 'user' ? 'bg-blue-600' : 'bg-gray-700'}`}>
                    <p>{msg.content}</p>
                    {msg.image && (
                      <img src={msg.image} alt="Generated" className="mt-2 rounded-lg max-w-full" />
                    )}
                  </div>
                </div>
              ))}

              {step1State === 'generating' && (
                <div className="flex justify-start">
                  <div className="bg-gray-700 rounded-lg p-3 flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-500 border-t-transparent" />
                    <span>Generating...</span>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="p-4 border-t border-gray-700">
              {error && (
                <div className="mb-2 p-2 bg-red-900/50 border border-red-700 rounded text-red-200 text-sm">
                  {error}
                </div>
              )}

              <div className="flex gap-2">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Describe your image..."
                  disabled={step1State === 'generating'}
                  className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 focus:outline-none focus:border-blue-500 disabled:opacity-50"
                />
                <button
                  onClick={handleSendMessage}
                  disabled={!input.trim() || step1State === 'generating'}
                  className="px-4 py-2 bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50"
                >
                  Send
                </button>
              </div>
            </div>
          </div>

          {/* Preview Panel */}
          <div className="w-1/2 flex flex-col">
            <div className="p-4 border-b border-gray-700">
              <h2 className="text-lg font-semibold">Preview</h2>
            </div>

            <div className="flex-1 flex flex-col items-center justify-center p-4">
              {currentImage ? (
                <>
                  <img src={currentImage} alt="Current" className="max-w-full max-h-[400px] rounded-lg shadow-lg" />
                  <p className="mt-4 text-gray-400 text-sm">
                    Happy with this image? Proceed to create 3D asset.
                  </p>
                  <button
                    onClick={handleProceedToStep2}
                    className="mt-4 px-6 py-3 bg-green-600 rounded-lg hover:bg-green-700 font-medium text-lg"
                  >
                    Proceed to 3D Generation →
                  </button>
                </>
              ) : (
                <p className="text-gray-500">Your generated image will appear here</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Step 2: Meshy 3D */}
      {currentStep === 2 && (
        <div className="flex h-[calc(100vh-73px)]">
          {/* Source Image Panel */}
          <div className="w-1/3 flex flex-col border-r border-gray-700">
            <div className="p-4 border-b border-gray-700">
              <h2 className="text-lg font-semibold">Source Image</h2>
              <button
                onClick={handleBackToStep1}
                className="text-sm text-blue-400 hover:text-blue-300 mt-1"
              >
                ← Back to image generation
              </button>
            </div>

            <div className="flex-1 flex items-center justify-center p-4">
              {currentImage && (
                <img src={currentImage} alt="Source" className="max-w-full max-h-full rounded-lg" />
              )}
            </div>
          </div>

          {/* 3D Preview Panel */}
          <div className="w-2/3 flex flex-col">
            <div className="p-4 border-b border-gray-700">
              <h2 className="text-lg font-semibold">Step 2: 3D Model</h2>
              <p className="text-gray-400 text-sm">Generate a 3D model from your image</p>
            </div>

            <div className="flex-1 flex flex-col items-center justify-center p-4">
              {error && (
                <div className="mb-4 p-3 bg-red-900/50 border border-red-700 rounded text-red-200">
                  {error}
                </div>
              )}

              {step2State === 'idle' && !glbUrl && (
                <div className="text-center">
                  <p className="text-gray-400 mb-4">Ready to generate 3D model</p>
                  <button
                    onClick={handleGenerate3D}
                    className="px-8 py-4 bg-green-600 rounded-lg hover:bg-green-700 font-medium text-xl"
                  >
                    Generate 3D Asset
                  </button>
                </div>
              )}

              {step2State === 'generating' && (
                <div className="text-center">
                  <div className="animate-spin rounded-full h-16 w-16 border-4 border-green-500 border-t-transparent mx-auto mb-4" />
                  <p className="text-xl">Generating 3D model...</p>
                  <p className="text-gray-400 text-lg mt-2">{progress}% complete</p>
                  <p className="text-gray-500 text-sm mt-2">This may take 1-2 minutes</p>
                </div>
              )}

              {step2State === 'saving' && (
                <div className="text-center">
                  <div className="animate-spin rounded-full h-16 w-16 border-4 border-green-500 border-t-transparent mx-auto mb-4" />
                  <p className="text-xl">Saving to library...</p>
                </div>
              )}

              {step2State === 'complete' && glbUrl && (
                <div className="w-full h-full flex flex-col">
                  <GLBViewer url={glbUrl} className="flex-1 min-h-[400px]" />
                  <div className="flex gap-4 justify-center mt-4">
                    <button
                      onClick={handleRegenerate}
                      className="px-6 py-3 bg-gray-600 rounded-lg hover:bg-gray-700"
                    >
                      Regenerate
                    </button>
                    <button
                      onClick={handleApproveAndSave}
                      className="px-6 py-3 bg-green-600 rounded-lg hover:bg-green-700"
                    >
                      Save to Library
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
