from ollama_client import ask_ollama, prepare_diagnosis


def main() -> None:
    raw_prediction = {'exercise': 'Standing Shoulder Internal/External Rotation',
 'movement_id': 7,
 'label': 0,
 'classification': 'correcta',
 'probability_incorrect': 0.4971,
 'threshold': 0.8809,
 'severity': 'moderada',
 'errors_detected': [],
 'biomechanical_summary': {'max_shoulder_angle_deg': 164.7449,
  'mean_shoulder_angle_deg': 164.7449,
  'trunk_tilt_p95_deg': 33.6901,
  'shoulder_elevation_p95': 0.0,
  'symmetry_p95_deg': 0.0,
  'velocity_std_deg': 0.0,
  'wrist_above_elbow_mean': 0,
  'valid_points_ratio_mean': 0.5}}

    diagnosis = prepare_diagnosis(raw_prediction)

    print("\n=== DIAGNOSIS ===\n")
    print(diagnosis)

    feedback = ask_ollama(raw_prediction)

    print("\n=== FEEDBACK ===\n")
    print(feedback)
    print("\n=== FEEDBACK ===\n")
    # print("Corrección prioritaria:", feedback["priority_correction"])
    # print("Feedback corto:", feedback["feedback_short"])
    # print("Feedback detallado:", feedback["feedback_detailed"])


if __name__ == "__main__":
    main()