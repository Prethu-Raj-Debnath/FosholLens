import { z } from 'zod';

export const CROPS = ['rice', 'potato', 'tomato', 'chili', 'unknown'] as const;
export const CropSchema = z.enum(CROPS);
export type Crop = z.infer<typeof CropSchema>;

export const CONFIDENCE_BANDS = ['low', 'medium', 'high'] as const;
export const ConfidenceBandSchema = z.enum(CONFIDENCE_BANDS);
export type ConfidenceBand = z.infer<typeof ConfidenceBandSchema>;

export const ApiErrorSchema = z.object({
  code: z.string(),
  message: z.string(),
  details: z.unknown().optional(),
});
export type ApiError = z.infer<typeof ApiErrorSchema>;

export const ApiErrorResponseSchema = z.object({
  error: ApiErrorSchema,
});
export type ApiErrorResponse = z.infer<typeof ApiErrorResponseSchema>;
