import { ACTION_MEANINGS, type ActionMeaning } from '../inference/types';

export type AleActionCode = 0 | 1 | 3 | 4;

export const ALE_ACTIONS: Record<ActionMeaning, AleActionCode> = {
  NOOP: 0,
  FIRE: 1,
  RIGHT: 3,
  LEFT: 4,
};

export interface MappedAction {
  modelIndex: number;
  meaning: ActionMeaning;
  aleAction: AleActionCode;
}

export function actionMeaningFromModelIndex(modelIndex: number): ActionMeaning {
  if (!Number.isInteger(modelIndex) || modelIndex < 0 || modelIndex >= ACTION_MEANINGS.length) {
    throw new Error(`model action index must be an integer from 0 to ${ACTION_MEANINGS.length - 1}`);
  }
  return ACTION_MEANINGS[modelIndex]!;
}

export function mapModelActionToAle(modelIndex: number): MappedAction {
  const meaning = actionMeaningFromModelIndex(modelIndex);
  return { modelIndex, meaning, aleAction: ALE_ACTIONS[meaning] };
}

export function validateMinimalActionSet(actionSet: readonly number[]): void {
  const actual = new Set(actionSet);
  const expected = new Set<AleActionCode>(Object.values(ALE_ACTIONS) as AleActionCode[]);
  const missing = [...expected].filter((action) => !actual.has(action));
  if (missing.length > 0) {
    throw new Error(`ALE action set is missing required action codes: ${missing.join(', ')}`);
  }
  const unexpected = [...actual].filter((action) => !expected.has(action as AleActionCode));
  if (unexpected.length > 0) {
    throw new Error(`ALE action set contains unexpected action codes: ${unexpected.join(', ')}`);
  }
}
