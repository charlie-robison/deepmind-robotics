'use client';

import { useEffect, useRef, useState } from 'react';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

export default function SplatViewer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const [loading, setLoading] = useState(true);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!containerRef.current || viewerRef.current) return;

    const viewer = new GaussianSplats3D.Viewer({
      cameraUp: [0, 1, 0],
      initialCameraPosition: [0, 2, 8],
      initialCameraLookAt: [0, 0, 0],
      sphericalHarmonicsDegree: 2,
      dynamicScene: false,
      rootElement: containerRef.current,
    });

    viewerRef.current = viewer;

    viewer.addSplatScene('/safety_park_output.ply', {
      splatAlphaRemovalThreshold: 5,
      showLoadingUI: false,
      progressiveLoad: true,
      onProgress: (percent: number, message: string) => {
        setProgress(Math.round(percent * 100));
      }
    })
    .then(() => {
      setLoading(false);
      viewer.start();
    })
    .catch((err: Error) => {
      setError(err.message);
      setLoading(false);
    });

    return () => {
      if (viewerRef.current) {
        viewerRef.current.dispose();
        viewerRef.current = null;
      }
    };
  }, []);

  return (
    <div className="relative w-full h-screen bg-black">
      <div ref={containerRef} className="w-full h-full" />

      {/* Info overlay */}
      <div className="absolute top-4 left-4 bg-black/80 text-white p-4 rounded-lg font-mono text-sm z-10">
        <strong>Gaussian Splat Viewer</strong>
        <br />
        Drag: rotate | Scroll: zoom | Right-drag: pan
      </div>

      {/* Loading overlay */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-20">
          <div className="text-white text-2xl font-mono text-center">
            <div>Loading Gaussian Splat...</div>
            <div className="text-4xl mt-2">{progress}%</div>
          </div>
        </div>
      )}

      {/* Error overlay */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-red-900/50 z-20">
          <div className="text-white text-xl font-mono text-center p-8">
            <div className="text-red-400">Error loading splat:</div>
            <div className="mt-2">{error}</div>
          </div>
        </div>
      )}
    </div>
  );
}
