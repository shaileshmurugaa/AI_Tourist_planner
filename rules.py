def apply_travel_rules(original_data, rf_pred, xgb_pred):
    # Example rule logic — replace with your actual rules
    final_pred = (rf_pred + xgb_pred) / 2

    # Example rule: adjust for senior travelers
    if original_data.get("Traveler age", 0) > 60:
        final_pred *= 0.95

    return final_pred
