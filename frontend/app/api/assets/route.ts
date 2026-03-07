import { listAssets } from '@/lib/assets';

export async function GET() {
  try {
    const assets = await listAssets();

    return Response.json({
      assets,
      total: assets.length,
    });
  } catch (error) {
    console.error('[Assets] Error listing assets:', error);
    return Response.json({ error: String(error) }, { status: 500 });
  }
}
