import { NextRequest } from 'next/server';
import { uploadImageToR2, uploadGLBToR2, generateFilename } from '@/lib/r2';

const MESHY_API_KEY = process.env.MESHY_API_KEY;
const MESHY_API_BASE = 'https://api.meshy.ai/openapi/v1';

export const maxDuration = 60;

// In-memory store for pending jobs (in production, use Redis/DB)
const pendingJobs = new Map<string, { status: string; glbUrl?: string; r2GlbUrl?: string; progress?: number }>();

// POST - Start a new Meshy job
export async function POST(req: NextRequest) {
  try {
    const { imageBase64 } = await req.json();

    if (!imageBase64) {
      return Response.json({ error: 'imageBase64 is required' }, { status: 400 });
    }

    if (!MESHY_API_KEY) {
      return Response.json({ error: 'MESHY_API_KEY not configured' }, { status: 500 });
    }

    // Upload image to R2 for a publicly accessible URL
    const filename = generateFilename('meshy-input', 'png');
    const uploadResult = await uploadImageToR2(imageBase64, filename);
    const imageUrl = uploadResult.publicUrl;
    console.log('[Meshy] Image uploaded to R2:', imageUrl);

    // Create Meshy task with texturing enabled
    const meshyRequest = {
      image_url: imageUrl,
      ai_model: 'meshy-4',
      topology: 'triangle',
      target_polycount: 30000,
      should_remesh: true,
      should_texture: true,  // Enable texture generation
      enable_pbr: true,      // Enable PBR materials for better lighting
    };

    const response = await fetch(`${MESHY_API_BASE}/image-to-3d`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${MESHY_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(meshyRequest),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('[Meshy] API error:', response.status, errorText);
      return Response.json({ error: `Meshy API error: ${response.status}` }, { status: response.status });
    }

    const result = await response.json();
    const taskId = result.result;

    console.log('[Meshy] Task created:', taskId);

    // Store job status
    pendingJobs.set(taskId, { status: 'pending', progress: 0 });

    return Response.json({
      taskId,
      status: 'pending',
      message: 'Meshy processing started',
      sourceImageKey: uploadResult.key,
    });

  } catch (error) {
    console.error('[Meshy] Error:', error);
    return Response.json({ error: String(error) }, { status: 500 });
  }
}

// GET - Poll job status
export async function GET(req: NextRequest) {
  const taskId = req.nextUrl.searchParams.get('taskId');

  if (!taskId) {
    return Response.json({ error: 'taskId is required' }, { status: 400 });
  }

  if (!MESHY_API_KEY) {
    return Response.json({ error: 'MESHY_API_KEY not configured' }, { status: 500 });
  }

  try {
    // Check Meshy API for task status
    const response = await fetch(`${MESHY_API_BASE}/image-to-3d/${taskId}`, {
      headers: {
        'Authorization': `Bearer ${MESHY_API_KEY}`,
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('[Meshy] Status check error:', response.status, errorText);
      return Response.json({ error: `Meshy API error: ${response.status}` }, { status: response.status });
    }

    const result = await response.json();

    console.log('[Meshy] Task status:', result.status, 'Progress:', result.progress);

    if (result.status === 'SUCCEEDED') {
      const meshyGlbUrl = result.model_urls?.glb;
      let r2GlbUrl: string | undefined;

      // Upload GLB to R2 for permanent storage
      if (meshyGlbUrl) {
        try {
          const glbFilename = generateFilename('model', 'glb');
          const glbUpload = await uploadGLBToR2(meshyGlbUrl, glbFilename);
          r2GlbUrl = glbUpload.publicUrl;
          console.log('[Meshy] GLB uploaded to R2:', r2GlbUrl);
        } catch (err) {
          console.error('[Meshy] R2 GLB upload failed:', err);
          // Continue with Meshy URL as fallback
        }
      }

      // Extract the key from the R2 URL
      const modelKey = r2GlbUrl ? r2GlbUrl.split('.r2.dev/')[1] : undefined;

      return Response.json({
        status: 'complete',
        glbUrl: r2GlbUrl || meshyGlbUrl,
        meshyGlbUrl: meshyGlbUrl,
        modelKey: modelKey,
        thumbnailUrl: result.thumbnail_url,
        progress: 100
      });
    } else if (result.status === 'FAILED') {
      return Response.json({
        status: 'failed',
        error: result.task_error?.message || 'Unknown error',
        progress: 0
      });
    } else {
      // PENDING or IN_PROGRESS
      return Response.json({
        status: 'processing',
        progress: result.progress || 0
      });
    }

  } catch (error) {
    console.error('[Meshy] Poll error:', error);
    return Response.json({ error: String(error) }, { status: 500 });
  }
}
