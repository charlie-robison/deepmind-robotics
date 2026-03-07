import { uploadImageToR2, generateFilename } from '@/lib/r2';

export const maxDuration = 60;

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL = 'gemini-2.0-flash-exp-image-generation';

interface GeminiImageResponse {
  candidates?: Array<{
    content?: {
      parts?: Array<{
        text?: string;
        inlineData?: {
          mimeType: string;
          data: string;
        };
      }>;
    };
    finishReason?: string;
  }>;
  promptFeedback?: {
    blockReason?: string;
  };
}

async function generateImage(prompt: string, existingImageBase64?: string): Promise<{ image?: string; text?: string; error?: string }> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`;

  const parts: Array<{ text?: string; inlineData?: { mimeType: string; data: string } }> = [
    { text: prompt }
  ];

  // If there's an existing image, include it for editing
  if (existingImageBase64) {
    const base64Data = existingImageBase64.replace(/^data:image\/\w+;base64,/, '');
    parts.push({
      inlineData: {
        mimeType: 'image/png',
        data: base64Data
      }
    });
  }

  const requestBody = {
    contents: [{ parts }],
    generationConfig: {
      responseModalities: ['IMAGE', 'TEXT']
    },
    safetySettings: [
      { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_NONE' },
      { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_NONE' },
      { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_NONE' },
      { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_NONE' }
    ]
  };

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error('Gemini API error:', response.status, errorText);
    return { error: `Gemini API error: ${response.status}` };
  }

  const result: GeminiImageResponse = await response.json();

  if (result.promptFeedback?.blockReason) {
    return { error: `Content blocked: ${result.promptFeedback.blockReason}` };
  }

  const candidate = result.candidates?.[0];
  if (!candidate?.content?.parts) {
    return { error: 'No response from Gemini' };
  }

  let imageData: string | undefined;
  let textResponse: string | undefined;

  for (const part of candidate.content.parts) {
    if (part.inlineData?.data) {
      imageData = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
    }
    if (part.text) {
      textResponse = part.text;
    }
  }

  return { image: imageData, text: textResponse };
}

export async function POST(req: Request) {
  const { messages, currentImage } = await req.json();

  const lastMessage = messages[messages.length - 1];
  const userPrompt = lastMessage?.content || '';

  // Generate image using Gemini
  const result = await generateImage(userPrompt, currentImage);

  if (result.error) {
    return Response.json({ error: result.error }, { status: 400 });
  }

  // Upload image to R2 if generated
  let r2Url: string | undefined;
  if (result.image) {
    try {
      const filename = generateFilename('gen', 'png');
      const uploadResult = await uploadImageToR2(result.image, filename);
      r2Url = uploadResult.publicUrl;
      console.log('[Chat] Image uploaded to R2:', r2Url);
    } catch (err) {
      console.error('[Chat] R2 upload failed:', err);
      // Continue without R2 URL - still return base64
    }
  }

  // Return the image and any text response
  return Response.json({
    image: result.image,
    imageUrl: r2Url,
    text: result.text || 'Here\'s your generated image!'
  });
}
