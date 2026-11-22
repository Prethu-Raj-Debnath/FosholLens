import { z } from 'zod';
import { ConfidenceBandSchema } from './common.js';

export const ANALYSIS_STATUSES = ['pending', 'processing', 'completed', 'failed'] as const;
export const AnalysisStatusSchema = z.enum(ANALYSIS_STATUSES);
export type AnalysisStatus = z.infer<typeof AnalysisStatusSchema>;

/**
 * Ordered pipeline stage. The loading UI shows a different line for each, which
 * is what stops a 10-second wait from being an anonymous spinner.
 */
export const ANALYSIS_STAGES = [
  'queued',
  'reading',
  'grounding',
  'writing',
  'speaking',
  'done',
] as const;
export const AnalysisStageSchema = z.enum(ANALYSIS_STAGES);
export type AnalysisStage = z.infer<typeof AnalysisStageSchema>;

export const AUDIO_STATUSES = ['none', 'pending', 'ready', 'failed'] as const;
export const AudioStatusSchema = z.enum(AUDIO_STATUSES);
export type AudioStatus = z.infer<typeof AudioStatusSchema>;

export const AudioInfoSchema = z.object({
  status: AudioStatusSchema,
  /** Relative API path, only present when status is 'ready'. */
  url: z.string().nullable().optional(),
  durationMs: z.number().int().nullable().optional(),
  byteSize: z.number().int().nullable().optional(),
});
export type AudioInfo = z.infer<typeof AudioInfoSchema>;

export const AnalysisSummarySchema = z.object({
  id: z.string().uuid(),
  createdAt: z.string().datetime(),
  cropName: z.string(),
  status: AnalysisStatusSchema,
  stage: AnalysisStageSchema,
  thumbnailUrl: z.string(),
  imageUrl: z.string(),
  diseaseLabel: z.string().nullable(),
  district: z.string().nullable().optional(),
});
export type AnalysisSummary = z.infer<typeof AnalysisSummarySchema>;

export const DiagnosisResultSchema = z.object({
  diseaseLabel: z.string(),
  cropName: z.string(),
  /** null when the engine cannot produce a reliable estimate. */
  confidence: z.number().min(0).max(1).nullable(),
  confidenceBand: ConfidenceBandSchema,
  isAiEstimate: z.literal(true),
});
export type DiagnosisResult = z.infer<typeof DiagnosisResultSchema>;

export const TreatmentSchema = z.object({
  summary: z.string(),
  steps: z.array(z.string()),
  /** Curated source, shown as a citation. */
  source: z.string().nullable().optional(),
});
export type Treatment = z.infer<typeof TreatmentSchema>;

export const RagSourceSchema = z.object({
  chunkId: z.string(),
  sourceTitle: z.string(),
  section: z.string(),
  diseaseId: z.string(),
  excerpt: z.string(),
  culturalOnly: z.boolean().default(false),
  referToOfficer: z.boolean().default(false),
});
export type RagSource = z.infer<typeof RagSourceSchema>;

export const AnalysisUncertaintySchema = z.object({
  isUncertain: z.boolean(),
  message: z.string(),
});
export type AnalysisUncertainty = z.infer<typeof AnalysisUncertaintySchema>;

export const WeatherSchema = z.object({
  district: z.string(),
  rainProbabilityNext12h: z.number().int().nullable().optional(),
  windSpeedMax12h: z.number().nullable().optional(),
  humidity: z.number().int().nullable().optional(),
  /** Decided in code, never by the language model. */
  sprayWindowOk: z.boolean(),
  reason: z.string(),
  source: z.string().optional(),
});
export type Weather = z.infer<typeof WeatherSchema>;

export const AnalysisResultSchema = z.object({
  id: z.string().uuid(),
  createdAt: z.string().datetime(),
  imageUrl: z.string(),
  cropName: z.string(),
  status: AnalysisStatusSchema,
  stage: AnalysisStageSchema,
  /** null when analysis failed or could not determine a confident diagnosis. */
  diagnosis: DiagnosisResultSchema.nullable(),
  symptoms: z.array(z.string()),
  treatment: TreatmentSchema.nullable(),
  prevention: z.array(z.string()),
  needsProfessionalVerification: z.boolean(),
  ragSources: z.array(RagSourceSchema).default([]),
  ppeReminder: z.string().nullable().optional(),
  referToOfficer: z.boolean().default(false),
  uncertainty: AnalysisUncertaintySchema,
  /** Short friendly Bangla text, shown in the UI and spoken by TTS. */
  narrative: z.string().default(''),
  /** 'template' means the LLM was unavailable and facts were composed directly. */
  generatedBy: z.enum(['llm', 'template']).default('template'),
  weather: WeatherSchema.nullable().optional(),
  audio: AudioInfoSchema.optional(),
  district: z.string().nullable().optional(),
});
export type AnalysisResult = z.infer<typeof AnalysisResultSchema>;
