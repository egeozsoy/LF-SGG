//
// Created by Ege on 13.10.22.
//

#ifndef SGG_MATCHER_BRANCHED_SSG_MATCHER_H
#define SGG_MATCHER_BRANCHED_SSG_MATCHER_H

#include <numeric>
#include <algorithm>
#include <memory>

using namespace std;

// Define enum for sub and obj
enum RelType {
    SUB = 0,
    OBJ = 1
};


class BranchedSSGMatcher {
public:
    const std::map<int, std::map<int, std::map<std::tuple<int, int, RelType>, int>>> collect_entities(const std::vector<std::tuple<int, int, int, int, int>> &triplets_with_instances) {
        std::map<int, std::map<int, std::map<std::tuple<int, int, RelType>, int>>> entities;
        for (const auto &triplet_with_instances: triplets_with_instances) {
            // Triplet consists of (sub_id, sub_instances, pred_id,obj_id,obj_instances). Assign these to variables.
            const auto &[sub_id, sub_instance, pred_id, obj_id, obj_instance] = triplet_with_instances;

            // Check if sub_id is already in entities. If not, create it.
            if (entities.find(sub_id) == entities.end()) {
                entities[sub_id] = {{sub_instance, {{std::make_tuple(pred_id, obj_id, SUB), 1}}}};
            } else {
                // Found
                // Check if sub_instance is already in entities[sub_id]. If not, create it.
                if (entities[sub_id].find(sub_instance) == entities[sub_id].end()) {
                    // Not found
                    entities[sub_id][sub_instance] = {{std::make_tuple(pred_id, obj_id, SUB), 1}};
                } else {
                    // Found
                    // Check if (pred_id,obj_id,"sub") is already in entities[sub_id][sub_instance]. If not, create it.
                    if (entities[sub_id][sub_instance].find(std::make_tuple(pred_id, obj_id, SUB)) == entities[sub_id][sub_instance].end()) {
                        // Not found
                        entities[sub_id][sub_instance][std::make_tuple(pred_id, obj_id, SUB)] = 1;
                    } else {
                        // Found
                        entities[sub_id][sub_instance][std::make_tuple(pred_id, obj_id, SUB)] += 1;
                    }
                }
            }
            // Check if obj_id is already in entities. If not, create it.
            if (entities.find(obj_id) == entities.end()) {
                // Not found
                entities[obj_id] = {{obj_instance, {{std::make_tuple(pred_id, sub_id, OBJ), 1}}}};
            } else {
                // Found
                // Check if obj_instance is already in entities[obj_id]. If not, create it.
                if (entities[obj_id].find(obj_instance) == entities[obj_id].end()) {
                    // Not found
                    entities[obj_id][obj_instance] = {{std::make_tuple(pred_id, sub_id, OBJ), 1}};
                } else {
                    // Found
                    // Check if (pred_id,sub_id,"obj") is already in entities[obj_id][obj_instance]. If not, create it.
                    if (entities[obj_id][obj_instance].find(std::make_tuple(pred_id, sub_id, OBJ)) == entities[obj_id][obj_instance].end()) {
                        // Not found
                        entities[obj_id][obj_instance][std::make_tuple(pred_id, sub_id, OBJ)] = 1;
                    } else {
                        // Found
                        entities[obj_id][obj_instance][std::make_tuple(pred_id, sub_id, OBJ)] += 1;
                    }
                }
            }
        }
        return entities;
    }

    const std::vector<std::tuple<int, int, int, int, int>>
    apply_mapping_to_predictions(const std::vector<std::tuple<int, int, int, int, int>> &predictions, const std::map<std::pair<int, int>, int> &mapping) {
        auto mapped_preds = std::vector<std::tuple<int, int, int, int, int>>();
        for (const auto &prediction: predictions) {
            // Prediction consists of (sub_id, sub_instances, pred_id,obj_id,obj_instances). Assign these to variables.
            auto [sub_id, sub_instance, pred_id, obj_id, obj_instance] = prediction;
            auto sub_tpl = std::make_pair(sub_id, sub_instance);
            auto obj_tpl = std::make_pair(obj_id, obj_instance);
            if (mapping.find(sub_tpl) != mapping.end()) {
                sub_instance = mapping.at(sub_tpl);
            } else {
                sub_instance = -1;
            }
            if (mapping.find(obj_tpl) != mapping.end()) {
                obj_instance = mapping.at(obj_tpl);
            } else {
                obj_instance = -1;
            }
            mapped_preds.push_back(std::make_tuple(sub_id, sub_instance, pred_id, obj_id, obj_instance));
        }
        return mapped_preds;
    }

    float calculate_global_recall(const std::vector<std::tuple<int, int, int, int, int>> &gts, const std::vector<std::tuple<int, int, int, int, int>> &preds) {
        auto gts_dict = std::map<std::tuple<int, int, int, int, int>, int>();
        auto preds_dict = std::map<std::tuple<int, int, int, int, int>, int>();

        for (const auto &gt: gts) {
            // GT consists of (sub_id, sub_instances, pred_id,obj_id,obj_instances). Assign these to variables.
            const auto &[sub_id, sub_instance, pred_id, obj_id, obj_instance] = gt;
            const auto tpl = std::make_tuple(sub_id, sub_instance, pred_id, obj_id, obj_instance);
            if (gts_dict.find(tpl) == gts_dict.end()) {
                gts_dict[tpl] = 1;
            } else {
                gts_dict[tpl] += 1;
            }
        }
        for (const auto &pred: preds) {
            // Prediction consists of (sub_id, sub_instances, pred_id,obj_id,obj_instances). Assign these to variables.
            const auto &[sub_id, sub_instance, pred_id, obj_id, obj_instance] = pred;
            const auto tpl = std::make_tuple(sub_id, sub_instance, pred_id, obj_id, obj_instance);
            if (preds_dict.find(tpl) == preds_dict.end()) {
                preds_dict[tpl] = 1;
            } else {
                preds_dict[tpl] += 1;
            }
        }

        auto correct_matches = 0.;
        for (const auto &gt_rel: gts_dict) {
            const auto &[gt_relation_tpl, gt_relation_count] = gt_rel;
            if (preds_dict.find(gt_relation_tpl) != preds_dict.end()) {
                correct_matches += std::min(gt_relation_count, preds_dict[gt_relation_tpl]);
            }
        }
        const auto recall = correct_matches / gts.size();

        return recall;

    }

    const std::map<std::pair<int, int>, int>
    branched_matching(const std::vector<std::vector<int>> &gts_vector, const std::vector<std::vector<int>> &preds_vector, int N = 3, const int depth_limit = 15, const bool allow_no_matching = false) {
        auto gts = std::vector<std::tuple<int, int, int, int, int>>();
        auto preds = std::vector<std::tuple<int, int, int, int, int>>();
        for (const auto &gt: gts_vector) {
            gts.emplace_back(gt[0], gt[1], gt[2], gt[3], gt[4]);
        }
        for (const auto &pred: preds_vector) {
            preds.emplace_back(pred[0], pred[1], pred[2], pred[3], pred[4]);
        }
        const auto ground_truth_entities = collect_entities(gts);
        auto prediction_entities_tmp = collect_entities(preds);

        if (allow_no_matching) {
            // Create enough dummy entity instances in predictions to match the number of entity instances in ground truth, allowing not matching. Do this for every entity
            for (const auto &gt_entity: ground_truth_entities) {
                const auto &[gt_entity_id, gt_entity_instances] = gt_entity;
                auto gt_entity_instance_count = gt_entity_instances.size();
                // find corresponding entity in predictions if exists, but has less instances than ground truth, add dummy instances
                if (prediction_entities_tmp.find(gt_entity_id) != prediction_entities_tmp.end()) {
                    auto &pred_entity_instances = prediction_entities_tmp[gt_entity_id];
                    auto pred_entity_instance_count = pred_entity_instances.size();
                    if (pred_entity_instance_count < gt_entity_instance_count) {
                        for (auto i = 0; i < gt_entity_instance_count - pred_entity_instance_count; i++) {
                            std::map<std::tuple<int, int, RelType>, int> dummy_instance_rels = {{std::tuple(-1, -1, SUB), -1}};
                            int dummy_instance_id = -i - 10;
                            pred_entity_instances[dummy_instance_id] = dummy_instance_rels;
                        }
                    }
                }
            }
        }
        const auto prediction_entities = prediction_entities_tmp;


        auto all_gt_instances = std::vector<std::tuple<int, int, int>>();
        for (const auto &gt_entity: ground_truth_entities) {
            for (const auto &gt_instance: gt_entity.second) {
                const int rel_sum = std::accumulate(gt_instance.second.begin(), gt_instance.second.end(), 0, [](auto prev_sum, auto &entry) { return prev_sum + entry.second; });
                all_gt_instances.emplace_back(gt_entity.first, gt_instance.first, rel_sum);
            }
        }
        // Sort all_gt_instances by the third element (rel_sum)
        std::sort(all_gt_instances.begin(), all_gt_instances.end(), [](auto &a, auto &b) { return std::get<2>(a) > std::get<2>(b); });

//        auto level_instance_id_mapping_and_pred_entities = std::vector<std::pair<std::map<std::pair<int, int>, int>, std::map<int, std::map<int, std::map<std::tuple<int, int, std::string>, int>>>>>();
        // Switching to a more efficient mapping representation: just a vector
        auto level_instance_id_mapping_and_pred_entities = std::vector<std::pair<std::vector<int>, std::map<int, std::map<int, std::map<std::tuple<int, int, RelType>, int>>>>>();
        level_instance_id_mapping_and_pred_entities.emplace_back(std::vector<int>(all_gt_instances.size(), -1), prediction_entities);

        int depth = 0;
        for (const auto &gt_tpl: all_gt_instances) {
            if (depth > depth_limit) {
                N = 1; // We can't just stop, but make it much faster
            }
            const auto &[entity_id, gt_instance, _] = gt_tpl;
            const auto &gt_rels = ground_truth_entities.at(entity_id).at(gt_instance);

//            auto next_level_instance_id_mapping_and_pred_entities = std::vector<std::pair<std::map<std::pair<int, int>, int>, std::map<int, std::map<int, std::map<std::tuple<int, int, std::string>, int>>>>>();
            auto next_level_instance_id_mapping_and_pred_entities = std::vector<std::pair<std::vector<int>, std::map<int, std::map<int, std::map<std::tuple<int, int, RelType>, int>>>>>();

            for (const auto &instance_id_mapping_and_pred_entities: level_instance_id_mapping_and_pred_entities) {
                const auto &[instance_id_mapping, pred_entities] = instance_id_mapping_and_pred_entities;
                if (pred_entities.find(entity_id) == pred_entities.end() || pred_entities.at(entity_id).empty()) { // if not
                    next_level_instance_id_mapping_and_pred_entities.emplace_back(instance_id_mapping, pred_entities);
                    continue;
                } else {
                    const auto &pred_same_entities = pred_entities.at(entity_id);
                    auto recalls = std::vector<float>();
                    auto best_pred_instances = std::vector<int>();
                    for (const auto &pred: pred_same_entities) {
                        const auto &[pred_instance, pred_rels] = pred;
                        if (pred_instance <= -10) { // dummy instance always has very low but non-zero recall
                            recalls.emplace_back(0.00001);
                            best_pred_instances.emplace_back(pred_instance);
                            continue;
                        }
                        auto correct_matches = 0.0;
                        for (const auto &gt_rel: gt_rels) {
                            const auto &[tpl, count] = gt_rel;
                            if (pred_rels.find(tpl) != pred_rels.end()) {
                                correct_matches += std::min(count, pred_rels.at(tpl));
                            }
                        }
                        const auto recall = correct_matches / std::accumulate(gt_rels.begin(), gt_rels.end(), 0.0, [](float sum, const auto &tpl) { return sum + std::get<1>(tpl); });
                        recalls.push_back(recall);
                        best_pred_instances.push_back(pred_instance);
                    }
                    // Sort best_pred_instances by recall, and take the top N
                    std::sort(best_pred_instances.begin(), best_pred_instances.end(), [&recalls](int i1, int i2) { return recalls[i1] > recalls[i2]; });

                    best_pred_instances.resize(std::min(N, (int) best_pred_instances.size()));
                    for (const auto &best_pred_instance: best_pred_instances) {
                        // Create a deepcopy of the instance_id_mapping
//                        tmp_instance_id_mapping[std::make_pair(entity_id, best_pred_instance)] = gt_instance;
                        auto tmp_instance_id_mapping = instance_id_mapping;
                        tmp_instance_id_mapping[depth] = best_pred_instance;
                        auto tmp_pred_entities = pred_entities;
                        tmp_pred_entities.at(entity_id).erase(best_pred_instance);
                        next_level_instance_id_mapping_and_pred_entities.emplace_back(tmp_instance_id_mapping, tmp_pred_entities);
                    }
                }
            }
            level_instance_id_mapping_and_pred_entities = std::move(next_level_instance_id_mapping_and_pred_entities);
            depth += 1;
        }

        auto best_matching_instance_id_mapping = std::map<std::pair<int, int>, int>();
        auto best_recall = 0.0;

        for (const auto &instance_id_mapping_and_pred_entities: level_instance_id_mapping_and_pred_entities) {
            const auto &[instance_id_mapping, pred_entities] = instance_id_mapping_and_pred_entities;
            // convert id _mapping to dict format
            auto instance_id_mapping_dict = std::map<std::pair<int, int>, int>();
            for (unsigned int i = 0; i < all_gt_instances.size(); i++) {
                const auto &[entity_id, gt_instance, _] = all_gt_instances[i];
                instance_id_mapping_dict[std::pair(entity_id, instance_id_mapping[i])] = gt_instance;
            }
            const auto mapped_preds = apply_mapping_to_predictions(preds, instance_id_mapping_dict);
            const auto recall = calculate_global_recall(gts, mapped_preds);
            if (recall > best_recall) {
                best_recall = recall;
                best_matching_instance_id_mapping = std::move(instance_id_mapping_dict);
            }
        }
        return best_matching_instance_id_mapping;
    }
};


#endif //SGG_MATCHER_BRANCHED_SSG_MATCHER_H
