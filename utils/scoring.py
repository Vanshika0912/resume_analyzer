# --------------------------------------------------------------
# Final Resume Score Calculation Module
# --------------------------------------------------------------

def calculate_final_score(skill_score, jd_score, keyword_score):
    """
    Calculate the final resume strength score based on weighted metrics.

    Weights:
    - Skills Match:        40%
    - JD Similarity:       40%
    - Keyword Match:       20%

    Returns:
        Score out of 100 (integer)
    """

    try:
        final = (
            (skill_score * 0.4) +
            (jd_score * 0.4) +
            (keyword_score * 0.2)
        )

        final = round(final, 2)

        # Ensure score remains between 0–100
        if final > 100:
            final = 100
        if final < 0:
            final = 0

        return final

    except Exception as e:
        print("Error in score calculation:", e)
        return 0
