import { S3Client, PutObjectCommand, GetObjectCommand } from '@aws-sdk/client-s3';

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

export interface UploadResult {
  key: string;
  publicUrl: string;
}

/**
 * Upload a base64 image to R2
 */
export async function uploadImageToR2(
  base64Data: string,
  filename: string
): Promise<UploadResult> {
  // Remove data URL prefix if present
  const base64Clean = base64Data.replace(/^data:image\/\w+;base64,/, '');
  const buffer = Buffer.from(base64Clean, 'base64');

  // Detect content type from data URL or default to png
  let contentType = 'image/png';
  const match = base64Data.match(/^data:(image\/\w+);base64,/);
  if (match) {
    contentType = match[1];
  }

  const key = `assets/images/${filename}`;

  await r2Client.send(
    new PutObjectCommand({
      Bucket: R2_BUCKET_NAME,
      Key: key,
      Body: buffer,
      ContentType: contentType,
    })
  );

  return {
    key,
    publicUrl: `${R2_PUBLIC_URL}/${key}`,
  };
}

/**
 * Upload a GLB model to R2 by fetching from a URL
 */
export async function uploadGLBToR2(
  sourceUrl: string,
  filename: string
): Promise<UploadResult> {
  // Fetch the GLB from the source URL
  const response = await fetch(sourceUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch GLB: ${response.status}`);
  }

  const buffer = Buffer.from(await response.arrayBuffer());
  const key = `assets/models/${filename}`;

  await r2Client.send(
    new PutObjectCommand({
      Bucket: R2_BUCKET_NAME,
      Key: key,
      Body: buffer,
      ContentType: 'model/gltf-binary',
    })
  );

  return {
    key,
    publicUrl: `${R2_PUBLIC_URL}/${key}`,
  };
}

/**
 * Generate a unique filename with timestamp
 */
export function generateFilename(prefix: string, extension: string): string {
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(2, 8);
  return `${prefix}-${timestamp}-${random}.${extension}`;
}
