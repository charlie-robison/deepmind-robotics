'use client';

import { useState, useEffect, use } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';

const GLBViewer = dynamic(() => import('@/components/GLBViewer'), { ssr: false });

interface Asset {
  id: string;
  name: string;
  imageUrl: string;
  modelUrl: string;
  createdAt: string;
}

interface PageProps {
  params: Promise<{ id: string }>;
}

export default function AssetDetailPage({ params }: PageProps) {
  const { id } = use(params);
  const [asset, setAsset] = useState<Asset | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAsset();
  }, [id]);

  const fetchAsset = async () => {
    try {
      const response = await fetch(`/api/assets/${id}`);
      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        setAsset(data.asset);
      }
    } catch (err) {
      setError('Failed to load asset');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (asset?.modelUrl) {
      window.open(asset.modelUrl, '_blank');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-2 border-blue-500 border-t-transparent" />
        <span className="ml-3 text-gray-400">Loading asset...</span>
      </div>
    );
  }

  if (error || !asset) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center">
        <div className="text-6xl mb-4">😕</div>
        <h2 className="text-xl font-semibold mb-2">Asset not found</h2>
        <p className="text-gray-400 mb-6">{error || 'This asset does not exist'}</p>
        <Link href="/library" className="text-blue-400 hover:text-blue-300">
          ← Back to Library
        </Link>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="border-b border-gray-700 p-6">
        <div className="max-w-6xl mx-auto">
          <Link href="/library" className="text-blue-400 hover:text-blue-300 text-sm mb-2 inline-block">
            ← Back to Library
          </Link>
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">{asset.name}</h1>
              <p className="text-gray-400 mt-1">
                Created {new Date(asset.createdAt).toLocaleString()}
              </p>
            </div>
            <button
              onClick={handleDownload}
              className="px-6 py-3 bg-green-600 rounded-lg hover:bg-green-700 font-medium"
            >
              Download GLB
            </button>
          </div>
        </div>
      </div>

      {/* Content - Side by side */}
      <div className="max-w-6xl mx-auto p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[calc(100vh-200px)]">
          {/* Source Image */}
          <div className="bg-gray-800 rounded-lg overflow-hidden border border-gray-700">
            <div className="p-4 border-b border-gray-700">
              <h2 className="font-semibold">Source Image</h2>
            </div>
            <div className="p-4 h-[calc(100%-60px)] flex items-center justify-center">
              <img
                src={asset.imageUrl}
                alt={asset.name}
                className="max-w-full max-h-full object-contain rounded-lg"
              />
            </div>
          </div>

          {/* 3D Model */}
          <div className="bg-gray-800 rounded-lg overflow-hidden border border-gray-700">
            <div className="p-4 border-b border-gray-700">
              <h2 className="font-semibold">3D Model</h2>
            </div>
            <div className="h-[calc(100%-60px)]">
              <GLBViewer url={asset.modelUrl} className="w-full h-full" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
