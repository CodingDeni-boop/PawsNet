from VideoRotator import VideoRotator
import os
from Dataset import VideoSequenceDataset, VideoSequenceDatasetNoLabels
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score
from Model import CNNTransformerClassifier
from model_methods import train_epoch, evaluate, predict_per_video
from post_processing import apply_gap_fill, apply_min_duration_filter, apply_supported_rearing_merge
from ModelWrapper import ModelWrapper
from plots import plot_instance_count_scatter
from create_video import annotate_video_with_predictions

if __name__ == "__main__":

    ROTATE = False                      #       <-
    EVALUATE_ON_TEST = True             #       <-  These three toggle certain parts of the code like switches.
    GENERATE_ANNOTATED_VIDEOS = True    #       <-
    RAW_VIDEOS_DIR = "./dataset/raw_videos"
    TRACKING_DIR = "./dataset/tracking"
    ROTATED_VIDEOS_DIR = "./dataset/rotated_videos"
    LABELS_DIR = "./dataset/labels"
    MODEL_PATH         = "model_saves/CNN_Transformer_for_evaluation.pth"
    PREDICTIONS_DIR      = "./predictions_for_evaluation"
    METRICS_DIR = "./metrics"
    SEQUENCE_LENGTH = 30
    TRAIN_STRIDE = 10                # For fast debugging put this to 5000
    EVAL_STRIDE     = 10             # For fast debugging put this to 2000
    TEST_STRIDE = 5                  # For fast debugging put this to 25
    IMG_SIZE        = (76, 142)
    BATCH_SIZE      = 32
    DROPOUT         = 0.3
    NUM_EPOCHS         = 150         # For fast debugging put this to 1
    LEARNING_RATE      = 0.0001
    CNN_FEATURE_DIM  = 512
    D_MODEL          = 512
    NHEAD            = 8
    NUM_LAYERS       = 3
    DIM_FEEDFORWARD  = 2048
    DROPOUT          = 0.3
    SMOOTHING = "gap"
    GAP_WINDOW = 5
    MIN_DURATION_WINDOW = 5
    CONFUSION_MATRIX_NORMALIZE = True
    COLUMN_NAMES = {0 : "background", 
                    1 : "supportedrear", 
                    2 : "unsupportedrear", 
                    3 : "grooming", 
                    4 : "digging"} #can be whatever you want, but column names need to be consistent in all labels df.

    TRAIN_VIDEO_NAMES = ['20231123_10min_OFT-BL_3961', '20231123_10min_OFT-BL_3962', '20231123_10min_OFT-BL_3963',
                        '20231123_10min_OFT-BL_3964', '20231123_10min_OFT-BL_4028',
                        '3278_21min_behaviour_2023-01-19T11_08_30', 'BehavioralCamera2023-02-14T13_05_19_shorter',
                        'BehavioralCamera2023-02-14T15_22_37_shorter', 'BehavioralCamera2023-02-15T14_40_46_shorter',
                        'BehavioralCamera2023-02-18T10_33_06_shorter', 'BehavioralCamera2023-02-18T12_37_43_shorter',
                        'BehavioralCamera2023-02-23T15_42_37_shorter', 'BehavioralCamera2023-03-09T10_37_32',
                        'BehavioralCamera2023-03-09T11_04_40', 'BehavioralCamera2023-03-09T11_41_07',
                        'BehavioralCamera2023-03-09T12_34_50', 'BehavioralCamera2023-03-09T13_02_04', 'MBT1-M10',
                        'MBT1-M11', 'MBT1-M15', 'MBT1-M18', 'MBT1-M2', 'MBT1-M6', 'T1', 'T12', 'T13', 'T14', 'T16',
                        'T17', 'T18', 'T19', 'T2', 'T5', 'T8', 'T9', 'OFT_left_1', 'OFT_left_2', 'OFT_left_3', 
                        'OFT_left_4', 'OFT_left_6', 'OFT_left_7', 'OFT_left_8', 'OFT_left_13', 'OFT_left_14', 'OFT_left_15', 
                        'OFT_left_16', 'OFT_left_17', 'OFT_left_18', 'OFT_left_19', 'OFT_left_20', 'OFT_left_21', ]# For fast debugging put this to ['T1']
    VALIDATION_VIDEO_NAMES = ['20231123_10min_OFT-BL_3919', '20231123_10min_OFT-BL_4029',
                        'BehavioralCamera2023-02-19T14_53_53_shorter', 'BehavioralCamera2023-03-09T14_30_45', 'MBT1-M14',
                        'MBT1-M3', 'T10', 'T3', 'T7', 'OFT_left_9', 'OFT_left_10' ]# For fast debugging put this to ['T3']
    TEST_VIDEO_NAMES = ['20231123_10min_OFT-BL_4025', '3279_21min_behaviour_2023-01-19T12_57_29',
                        'BehavioralCamera2023-02-23T10_23_42_shorter', 'BehavioralCamera2023-02-24T11_06_53_shorter',
                        'BehavioralCamera2023-03-09T12_08_14', 'MBT1-M7', 'T11', 'T15', 'T4', 'T6', 'OFT_left_11', 'OFT_left_12' ]# For fast debugging put this to ['T4']

    behaviors = list(COLUMN_NAMES.values())

    ###                 ACTUAL CODE                     ###

    if ROTATE:
        for video_name in os.listdir(RAW_VIDEOS_DIR):
            if not (video_name.endswith(".mp4") or video_name.endswith(".avi")):
                print(f"{video_name} is not a valid video file and wasn't rotated")
                continue            ## If not .mp4 or .avi file, it is not rotated
            
            name, extension = os.path.splitext(video_name)

            video_path = os.path.join(RAW_VIDEOS_DIR, video_name)
            
            tracking_name = name + ".csv"
            tracking_path = os.path.join(TRACKING_DIR, tracking_name)

            output_name = name + ".mp4"
            output_path = os.path.join(ROTATED_VIDEOS_DIR, output_name)

            rotator = VideoRotator(video_path = video_path, output_path = output_path, out_width = 76, out_height = 142)
            rotator.follow(tracking_path, "mouse_top.mouse_top_0.bodycentre","mouse_top.mouse_top_0.neck")



    print(f"Split: {len(TRAIN_VIDEO_NAMES)} train / {len(VALIDATION_VIDEO_NAMES)} val / {len(TEST_VIDEO_NAMES)} test videos")
    print(f"Val IDs:  {VALIDATION_VIDEO_NAMES}")
    print(f"Test IDs: {TEST_VIDEO_NAMES}")

    device = torch.device("cuda" if torch.cuda.is_available() else
                            "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = VideoSequenceDataset(
        ROTATED_VIDEOS_DIR, LABELS_DIR, TRAIN_VIDEO_NAMES,
        SEQUENCE_LENGTH, TRAIN_STRIDE, IMG_SIZE,
        behavior_names=behaviors,
        augment=True   # CHANGE: augmentation enabled for training
    )
    val_dataset = VideoSequenceDataset(
        ROTATED_VIDEOS_DIR, LABELS_DIR, VALIDATION_VIDEO_NAMES,
        SEQUENCE_LENGTH, EVAL_STRIDE, IMG_SIZE,
        behavior_names=behaviors,
        augment=False
    )
    print(f"Train eval dataset: {len(train_dataset)} sequences")
    print(f"Val eval dataset:   {len(val_dataset)} sequences")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=2, pin_memory=True)

    model = CNNTransformerClassifier(
        cnn_feature_dim=CNN_FEATURE_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        num_classes=len(behaviors),
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)

    print(f"\nModel architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


    ###         Here it's about finding the parameter weight_tensor for each behavior. 
    ###         It's a really complicated (and probably superfluous) calculation based on class distribution.
    ###         See report methods for more infos. Would want to change this in future.

    all_train_labels = np.concatenate([
        train_dataset.label_cache[vid] for vid in train_dataset.label_cache
    ])
    unique, counts = np.unique(all_train_labels, return_counts=True)
    class_counts   = dict(zip(unique, counts))
    total_samples  = len(all_train_labels)

    print(f"\n=== Class Distribution ===")
    for cls_idx, cls_name in enumerate(behaviors):
        count      = class_counts.get(cls_idx, 0)
        percentage = 100 * count / total_samples
        print(f"{cls_name}: {count} ({percentage:.2f}%)")

    class_weights = {}
    for cls_idx in range(len(behaviors)):
        count = class_counts.get(cls_idx, 0)
        class_weights[cls_idx] = (total_samples / (len(behaviors) * count)) ** 1.0 if count > 0 else 1.0

    # CHANGE (v20): Boost underperforming classes before capping background.
    # Unsupportedrearing and Grooming consistently spill into background on test set.
    CLASS_BOOSTS = {'Unsupportedrearing': 1.5, 'Grooming': 1.5}
    for cls_idx, cls_name in enumerate(behaviors):
        if cls_name in CLASS_BOOSTS:
            class_weights[cls_idx] *= CLASS_BOOSTS[cls_name]

    # CHANGE (v20): Background weight cap lowered from 0.5 to 0.2.
    # With ~74% background frames and 5 classes, the natural inverse-frequency weight
    # is ~0.27 — the old cap of 0.5 was above that, so it had no effect at all.
    background_idx = next((i for i, n in enumerate(behaviors) if n.lower() == 'background'), None)
    if background_idx is not None:
        class_weights[background_idx] = min(class_weights[background_idx], 0.2)

    class_weights_array = np.array([class_weights[i] for i in range(len(behaviors))])

    print(f"\n=== Class Weights (power=1.0, boosts applied, background capped at 0.2) ===")
    for cls_idx, cls_name in enumerate(behaviors):
        boost_str = f" [x{CLASS_BOOSTS[cls_name]}]" if cls_name in CLASS_BOOSTS else ""
        print(f"{cls_name}: {class_weights_array[cls_idx]:.3f} (count: {class_counts.get(cls_idx, 0)}){boost_str}")

    weight_tensor = torch.FloatTensor(class_weights_array).to(device)

    ###  weight_tensor now found. This whole code part could be substituted by arbitrarily chosing class_weights = [1, 2, 2.5, 2, 1]. 

    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    best_f1          = 0.0
    patience         = 15
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        y_pred, y_true = evaluate(model, val_loader, device, background_class = 0)
        val_acc = 100 * np.sum(y_pred == y_true) / len(y_true)
        val_f1  = f1_score(y_true, y_pred, average='macro')

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Acc (consensus): {val_acc:.2f}%, Val F1 (consensus): {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'cnn_feature_dim':  CNN_FEATURE_DIM,
                'd_model':          D_MODEL,
                'nhead':            NHEAD,
                'num_layers':       NUM_LAYERS,
                'dim_feedforward':  DIM_FEEDFORWARD,
                'num_classes':      len(behaviors),
                'sequence_length':  SEQUENCE_LENGTH,
                'img_size':         IMG_SIZE,
                'dropout':          DROPOUT,
            }, MODEL_PATH)
            print(f"→ New best model saved! Val F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    ### COMPUTE PREDICTIONS ON THE TEST SET AND SAVE THEM AS CSV, AND PLOT GRAPHS
    if EVALUATE_ON_TEST:

        print("\nLoading best model...")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_dataset = VideoSequenceDatasetNoLabels(ROTATED_VIDEOS_DIR, TEST_VIDEO_NAMES, SEQUENCE_LENGTH, TEST_STRIDE, IMG_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        per_video = predict_per_video(model, test_loader, device)
        for video_id, (frame_indices, preds) in per_video.items():
            preds = apply_gap_fill(preds)
            preds = apply_min_duration_filter(preds)
            one_hot = np.zeros((len(preds), len(behaviors)), dtype=int)
            for i, cls_idx in enumerate(preds):
                one_hot[i, cls_idx] = 1

            df = pd.DataFrame(one_hot, columns=behaviors, index=frame_indices)
            df.index.name = "frame"

            out_path = os.path.join(PREDICTIONS_DIR, f"{video_id}.csv")
            df.to_csv(out_path)
            print(f"Saved: {out_path}  ({len(df)} frames)")
        
        CNNTransformer = ModelWrapper(name = "CNN-Transformer", test_set = TEST_VIDEO_NAMES, predictions_folder = PREDICTIONS_DIR, true_folder = LABELS_DIR,
                                column_names = COLUMN_NAMES, output_folder = METRICS_DIR, smoothing = SMOOTHING, gap_window = GAP_WINDOW, 
                                min_duration_window = MIN_DURATION_WINDOW)
        CNNTransformer.plot_confusion_matrix(normalize = CONFUSION_MATRIX_NORMALIZE)
        plot_instance_count_scatter(model_wrappers=[CNNTransformer],output_path=os.path.join(METRICS_DIR, "behavior_instance_count.png"))

        for i, video_id in enumerate(TEST_VIDEO_NAMES):
            annotate_video_with_predictions(
                video_path = os.path.join(ROTATED_VIDEOS_DIR, f"{video_id}.mp4"),
                predictions = CNNTransformer.label_wrappers[i].pred,
                output_path = os.path.join(METRICS_DIR, f"{video_id}_annotated_by_{CNNTransformer.name}.mp4"),
                true_labels = CNNTransformer.label_wrappers[i].true,
                column_names = COLUMN_NAMES
            )