import { ACTION_MEANINGS, type ActionMeaning, type QMarginDiagnostics } from '../inference/types';
import type { BrowserFixtureReference } from './fixtures';

export interface FixtureCalculation {
  maxAbsoluteError: number;
  meanAbsoluteError: number;
  actionAgreementRate: number;
  disagreementIndices: number[];
  qMargin: QMarginDiagnostics;
  passed: boolean;
}

function actionForIndex(index: number): ActionMeaning {
  const action = ACTION_MEANINGS[index];
  if (!action) throw new Error(`invalid action index ${index}`);
  return action;
}

function percentile(values: readonly number[], probability: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const position = (sorted.length - 1) * probability;
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  const lower = sorted[lowerIndex]!;
  const upper = sorted[upperIndex]!;
  return lower + (upper - lower) * (position - lowerIndex);
}

function qMargin(values: readonly number[]): number {
  if (values.length !== ACTION_MEANINGS.length || values.some((value) => !Number.isFinite(value))) {
    throw new Error('Q-margin calculation requires four finite Q-values');
  }
  const sorted = [...values].sort((left, right) => right - left);
  return sorted[0]! - sorted[1]!;
}

function buildQMarginDiagnostics(
  predictedMargins: readonly number[],
  referenceMargins: readonly number[],
): QMarginDiagnostics {
  const absoluteMarginErrors = predictedMargins.map((margin, index) => Math.abs(margin - referenceMargins[index]!));
  return {
    minMargin: Math.min(...predictedMargins),
    meanMargin: predictedMargins.reduce((sum, value) => sum + value, 0) / predictedMargins.length,
    p50Margin: percentile(predictedMargins, 0.5),
    maxMargin: Math.max(...predictedMargins),
    referenceMinMargin: Math.min(...referenceMargins),
    referenceMeanMargin: referenceMargins.reduce((sum, value) => sum + value, 0) / referenceMargins.length,
    referenceP50Margin: percentile(referenceMargins, 0.5),
    referenceMaxMargin: Math.max(...referenceMargins),
    maxAbsoluteMarginError: Math.max(...absoluteMarginErrors),
  };
}

export function calculateFixtureValidation(
  predictedQValues: readonly (readonly number[])[],
  predictedActions: readonly number[],
  reference: BrowserFixtureReference,
): FixtureCalculation {
  if (predictedQValues.length !== reference.sample_count || predictedActions.length !== reference.sample_count) {
    throw new Error('prediction row count does not match fixture sample_count');
  }

  let absoluteErrorSum = 0;
  let comparedValues = 0;
  let maxAbsoluteError = 0;
  let actionMatches = 0;
  const disagreementIndices: number[] = [];
  const predictedMargins: number[] = [];
  const referenceMargins: number[] = [];

  for (let sampleIndex = 0; sampleIndex < reference.sample_count; sampleIndex += 1) {
    const predicted = predictedQValues[sampleIndex];
    const expected = reference.q_values[sampleIndex];
    if (!predicted || !expected || predicted.length !== ACTION_MEANINGS.length) {
      throw new Error(`prediction at sample ${sampleIndex} must contain four Q-values`);
    }
    predictedMargins.push(qMargin(predicted));
    referenceMargins.push(qMargin(expected));
    for (let actionIndex = 0; actionIndex < ACTION_MEANINGS.length; actionIndex += 1) {
      const predictedValue = predicted[actionIndex]!;
      const expectedValue = expected[actionIndex]!;
      if (!Number.isFinite(predictedValue) || !Number.isFinite(expectedValue)) {
        throw new Error(`non-finite Q-value comparison at sample ${sampleIndex}, action ${actionIndex}`);
      }
      const error = Math.abs(predictedValue - expectedValue);
      absoluteErrorSum += error;
      comparedValues += 1;
      maxAbsoluteError = Math.max(maxAbsoluteError, error);
    }

    const predictedAction = predictedActions[sampleIndex]!;
    actionForIndex(predictedAction);
    if (predictedAction === reference.greedy_actions[sampleIndex]) {
      actionMatches += 1;
    } else {
      disagreementIndices.push(sampleIndex);
    }
  }

  const meanAbsoluteError = comparedValues === 0 ? 0 : absoluteErrorSum / comparedValues;
  const actionAgreementRate = reference.sample_count === 0 ? 0 : actionMatches / reference.sample_count;
  const passed =
    maxAbsoluteError <= reference.thresholds.max_absolute_error &&
    meanAbsoluteError <= reference.thresholds.mean_absolute_error &&
    actionAgreementRate >= reference.thresholds.action_agreement_rate;

  return {
    maxAbsoluteError,
    meanAbsoluteError,
    actionAgreementRate,
    disagreementIndices,
    qMargin: buildQMarginDiagnostics(predictedMargins, referenceMargins),
    passed,
  };
}
