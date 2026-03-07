import { S3Client, PutObjectCommand, ListObjectsV2Command, GetObjectCommand } from '@aws-sdk/client-s3';

const R2_ACCOUNT_ID = process.env.R2_ACCOUNT_ID!;
const R2_ACCESS_KEY_ID = process.env.R2_ACCESS_KEY_ID!;
const R2_SECRET_ACCESS_KEY = process.env.R2_SECRET_ACCESS_KEY!;
const R2_BUCKET_NAME = process.env.R2_BUCKET_NAME!;
const R2_PUBLIC_URL = process.env.R2_PUBLIC_URL!;

const r2Client = new S3Client({
  region: 'auto',
  endpoint: `https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
  credentials: {
    accessKeyId: R2_ACCESS_KEY_ID,
    secretAccessKey: R2_SECRET_ACCESS_KEY,
  },
});

export interface Asset {
  id: string;
  name: string;
  imageUrl: string;
  modelUrl: string;
  createdAt: string;
}

export interface AssetMetadata {
  id: string;
  name: string;
  sourceImageKey: string;
  modelKey: string;
  createdAt: string;
}

/**
 * Save asset metadata to R2
 */
export async function saveAssetMetadata(metadata: AssetMetadata): Promise<void> {
  const key = `assets/metadata/${metadata.id}.json`;

  await r2Client.send(
    new PutObjectCommand({
      Bucket: R2_BUCKET_NAME,
      Key: key,
      Body: JSON.stringify(metadata),
      ContentType: 'application/json',
    })
  );
}

/**
 * List all assets with their metadata
 */
export async function listAssets(): Promise<Asset[]> {
  const response = await r2Client.send(
    new ListObjectsV2Command({
      Bucket: R2_BUCKET_NAME,
      Prefix: 'assets/metadata/',
      MaxKeys: 100,
    })
  );

  const assets: Asset[] = [];

  for (const item of response.Contents || []) {
    if (!item.Key?.endsWith('.json')) continue;

    try {
      const getResponse = await r2Client.send(
        new GetObjectCommand({
          Bucket: R2_BUCKET_NAME,
          Key: item.Key,
        })
      );

      const body = await getResponse.Body?.transformToString();
      if (body) {
        const metadata: AssetMetadata = JSON.parse(body);
        assets.push({
          id: metadata.id,
          name: metadata.name,
          imageUrl: `${R2_PUBLIC_URL}/${metadata.sourceImageKey}`,
          modelUrl: `${R2_PUBLIC_URL}/${metadata.modelKey}`,
          createdAt: metadata.createdAt,
        });
      }
    } catch (err) {
      console.error('Error loading asset metadata:', item.Key, err);
    }
  }

  // Sort by createdAt descending (newest first)
  assets.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());

  return assets;
}

/**
 * Get a single asset by ID
 */
export async function getAsset(id: string): Promise<Asset | null> {
  try {
    const key = `assets/metadata/${id}.json`;

    const response = await r2Client.send(
      new GetObjectCommand({
        Bucket: R2_BUCKET_NAME,
        Key: key,
      })
    );

    const body = await response.Body?.transformToString();
    if (!body) return null;

    const metadata: AssetMetadata = JSON.parse(body);

    return {
      id: metadata.id,
      name: metadata.name,
      imageUrl: `${R2_PUBLIC_URL}/${metadata.sourceImageKey}`,
      modelUrl: `${R2_PUBLIC_URL}/${metadata.modelKey}`,
      createdAt: metadata.createdAt,
    };
  } catch (err) {
    console.error('Error getting asset:', id, err);
    return null;
  }
}
