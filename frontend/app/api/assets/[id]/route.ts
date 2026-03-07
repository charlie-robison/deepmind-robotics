import { getAsset } from '@/lib/assets';
import { NextRequest } from 'next/server';

interface RouteParams {
  params: Promise<{ id: string }>;
}

export async function GET(req: NextRequest, { params }: RouteParams) {
  try {
    const { id } = await params;

    const asset = await getAsset(id);

    if (!asset) {
      return Response.json({ error: 'Asset not found' }, { status: 404 });
    }

    return Response.json({ asset });
  } catch (error) {
    console.error('[Asset] Error:', error);
    return Response.json({ error: String(error) }, { status: 500 });
  }
}
