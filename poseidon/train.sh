#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export WANDB_MODE=disabled
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"

accelerate launch "${SCRIPT_DIR}/scOT/train.py" \
  --config "${SCRIPT_DIR}/configs/run_small.yaml" \
  --wandb_run_name "se-af-scratch-small" \
  --wandb_project_name "scOT" \
  --checkpoint_path "${REPO_ROOT}/tmp_checkpoints" \
  --data_path "${SCRIPT_DIR}/../dataset"


# Model size: 20742962
# Model size without embeddings: 20741256
# {'loss': 0.9656, 'grad_norm': 146.7209930419922, 'learning_rate': 4.999811754597862e-05, 'epoch': 0.08}                                                                          
# {'loss': 0.8525, 'grad_norm': 134.25721740722656, 'learning_rate': 4.9992470467405104e-05, 'epoch': 0.16}                                                                        
# {'loss': 0.7787, 'grad_norm': 174.00440979003906, 'learning_rate': 4.998305961470874e-05, 'epoch': 0.23}                                                                         
# {'loss': 0.6948, 'grad_norm': 92.70581817626953, 'learning_rate': 4.996988640512931e-05, 'epoch': 0.31}                                                                          
# {'loss': 0.6806, 'grad_norm': 329.88519287109375, 'learning_rate': 4.995295282250373e-05, 'epoch': 0.39}                                                                         
# {'loss': 0.6401, 'grad_norm': 83.65001678466797, 'learning_rate': 4.993226141696726e-05, 'epoch': 0.47}                                                                          
# {'loss': 0.6359, 'grad_norm': 75.49742889404297, 'learning_rate': 4.990781530456945e-05, 'epoch': 0.55}                                                                          
# {'loss': 0.631, 'grad_norm': 55.52005386352539, 'learning_rate': 4.987961816680492e-05, 'epoch': 0.62}                                                                           
# {'loss': 0.578, 'grad_norm': 265.07525634765625, 'learning_rate': 4.984767425005891e-05, 'epoch': 0.7}                                                                           
# {'loss': 0.568, 'grad_norm': 206.4291229248047, 'learning_rate': 4.981198836496775e-05, 'epoch': 0.78}                                                                           
# {'loss': 0.6197, 'grad_norm': 62.89774703979492, 'learning_rate': 4.97725658856945e-05, 'epoch': 0.86}                                                                           
# {'loss': 0.5696, 'grad_norm': 49.3708381652832, 'learning_rate': 4.972941274911953e-05, 'epoch': 0.94}                                                                           
# {'eval_loss': 0.5484529137611389, 'eval_median_relative_l1_error': 54.99919128417969, 'eval_mean_relative_l1_error': 54.78719711303711, 'eval_std_relative_l1_error': 6.149773597717285, 'eval_min_relative_l1_error': 40.44014358520508, 'eval_max_relative_l1_error': 78.98898315429688, 'eval_runtime': 2.9065, 'eval_samples_per_second': 41.287, 'eval_steps_per_second': 20.644, 'epoch': 1.0}
# {'loss': 0.5666, 'grad_norm': 129.9266357421875, 'learning_rate': 4.9682535453946464e-05, 'epoch': 1.02}                                                                         
# {'loss': 0.544, 'grad_norm': 98.63272094726562, 'learning_rate': 4.963194105972353e-05, 'epoch': 1.09}                                                                           
# {'loss': 0.5588, 'grad_norm': 87.63915252685547, 'learning_rate': 4.957763718578041e-05, 'epoch': 1.17}                                                                          
# {'loss': 0.5541, 'grad_norm': 127.03975677490234, 'learning_rate': 4.951963201008076e-05, 'epoch': 1.25}                                                                         
# {'loss': 0.5691, 'grad_norm': 151.7929229736328, 'learning_rate': 4.9457934267990693e-05, 'epoch': 1.33}                                                                         
# {'loss': 0.5584, 'grad_norm': 260.6266784667969, 'learning_rate': 4.9392553250963215e-05, 'epoch': 1.41}                                                                         
# {'loss': 0.5166, 'grad_norm': 59.19145965576172, 'learning_rate': 4.932349880513901e-05, 'epoch': 1.48}                                                                          
# {'loss': 0.5102, 'grad_norm': 94.78648376464844, 'learning_rate': 4.9250781329863606e-05, 'epoch': 1.56}                                                                         
# {'loss': 0.5606, 'grad_norm': 94.97967529296875, 'learning_rate': 4.9174411776121305e-05, 'epoch': 1.64}                                                                         
# {'loss': 0.4816, 'grad_norm': 39.779579162597656, 'learning_rate': 4.9094401644886e-05, 'epoch': 1.72}                                                                           
# {'loss': 0.5059, 'grad_norm': 131.793212890625, 'learning_rate': 4.901076298538915e-05, 'epoch': 1.8}                                                                            
# {'loss': 0.5081, 'grad_norm': 146.22848510742188, 'learning_rate': 4.892350839330522e-05, 'epoch': 1.88}                                                                         
# {'loss': 0.4847, 'grad_norm': 88.47186279296875, 'learning_rate': 4.8832651008854845e-05, 'epoch': 1.95}                                                                         
# {'eval_loss': 0.4875751733779907, 'eval_median_relative_l1_error': 47.922855377197266, 'eval_mean_relative_l1_error': 48.87066650390625, 'eval_std_relative_l1_error': 6.063055515289307, 'eval_min_relative_l1_error': 34.54719924926758, 'eval_max_relative_l1_error': 71.6068115234375, 'eval_runtime': 3.1331, 'eval_samples_per_second': 38.301, 'eval_steps_per_second': 19.151, 'epoch': 2.0}