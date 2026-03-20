from ollama_client import ask_ollama, prepare_diagnosis


def main() -> None:
    raw_prediction = {'exercise': 'Standing Shoulder Abduction',
 'movement_id': 9,
 'label': 1,
 'classification': 'incorrecta',
 'probability_incorrect': 1.0,
 'threshold': 0.9398,
 'severity': 'severa',
 'errors_detected': ['inclinacion_excesiva_tronco',
  'descenso_de_codo_o_muneca'],
 'biomechanical_summary': {'max_shoulder_angle_deg': 164.7449,
  'mean_shoulder_angle_deg': 164.7449,
  'trunk_tilt_p95_deg': 33.6901,
  'shoulder_elevation_p95': 0.0,
  'symmetry_p95_deg': 0.0,
  'velocity_std_deg': 0.0,
  'wrist_above_elbow_mean': -0.4743,
  'valid_points_ratio_mean': 0.6667}}

    diagnosis = prepare_diagnosis(raw_prediction)

    print("\n=== DIAGNOSIS ===\n")
    print(diagnosis)

    feedback = ask_ollama(raw_prediction)

    print("\n=== FEEDBACK ===\n")
    print("Corrección prioritaria:", feedback["priority_correction"])
    print("Feedback corto:", feedback["feedback_short"])
    print("Feedback detallado:", feedback["feedback_detailed"])


if __name__ == "__main__":
    main()