import { saveAssetMetadata, AssetMetadata } from '@/lib/assets';

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { name, sourceImageKey, modelKey } = body;

    if (!sourceImageKey || !modelKey) {
      return Response.json({ error: 'sourceImageKey and modelKey are required' }, { status: 400 });
    }

    const id = `asset-${Date.now()}-${Math.random().toString(36).substring(2, 8)}`;

    const metadata: AssetMetadata = {
      id,
      name: name || `Asset ${new Date().toLocaleDateString()}`,
      sourceImageKey,
      modelKey,
      createdAt: new Date().toISOString(),
    };

    await saveAssetMetadata(metadata);

    return Response.json({ success: true, id, metadata });
  } catch (error) {
    console.error('[Assets Save] Error:', error);
    return Response.json({ error: String(error) }, { status: 500 });
  }
}
