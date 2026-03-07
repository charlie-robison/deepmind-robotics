'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

interface Asset {
  id: string;
  name: string;
  imageUrl: string;
  modelUrl: string;
  createdAt: string;
}

export default function LibraryPage() {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAssets();
  }, []);

  const fetchAssets = async () => {
    try {
      const response = await fetch('/api/assets');
      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        setAssets(data.assets || []);
      }
    } catch (err) {
      setError('Failed to load assets');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="border-b border-gray-700 p-6">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Asset Library</h1>
            <p className="text-gray-400 mt-1">Your generated 3D assets</p>
          </div>
          <Link
            href="/create-asset"
            className="px-6 py-3 bg-green-600 rounded-lg hover:bg-green-700 font-medium"
          >
            + Create Asset
          </Link>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-6xl mx-auto p-6">
        {loading && (
          <div className="flex items-center justify-center py-20">
            <div className="animate-spin rounded-full h-8 w-8 border-2 border-blue-500 border-t-transparent" />
            <span className="ml-3 text-gray-400">Loading assets...</span>
          </div>
        )}

        {error && (
          <div className="p-4 bg-red-900/50 border border-red-700 rounded-lg text-red-200">
            {error}
          </div>
        )}

        {!loading && !error && assets.length === 0 && (
          <div className="text-center py-20">
            <div className="text-6xl mb-4">📦</div>
            <h2 className="text-xl font-semibold mb-2">No assets yet</h2>
            <p className="text-gray-400 mb-6">Create your first 3D asset to get started</p>
            <Link
              href="/create-asset"
              className="inline-block px-6 py-3 bg-green-600 rounded-lg hover:bg-green-700 font-medium"
            >
              Create Asset
            </Link>
          </div>
        )}

        {!loading && assets.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {assets.map((asset) => (
              <Link
                key={asset.id}
                href={`/library/${asset.id}`}
                className="group bg-gray-800 rounded-lg overflow-hidden border border-gray-700 hover:border-blue-500 transition-colors"
              >
                <div className="aspect-square relative bg-gray-900">
                  <img
                    src={asset.imageUrl}
                    alt={asset.name}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                    <span className="text-white font-medium">View Details</span>
                  </div>
                </div>
                <div className="p-4">
                  <h3 className="font-medium truncate">{asset.name}</h3>
                  <p className="text-sm text-gray-400 mt-1">
                    {new Date(asset.createdAt).toLocaleDateString()}
                  </p>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
