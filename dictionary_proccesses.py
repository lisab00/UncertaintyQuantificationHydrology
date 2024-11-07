
bounds_dict_snw = {
        'snw_dth': (-9999,-9999),
        'snw_ast': (-9999, -9999),
        'snw_amt': (-9999, -9999),
        'snw_amf': (-9999, -9999),
        'snw_pmf': (-9999, -9999),

        'sl0_mse': (0.00, 1e+2),
        'sl1_mse': (0.00, 2e+2),

        'sl0_fcy': (0.00, 2e+2),
        'sl0_bt0': (0.00, 3.00),

        'sl1_pwp': (0.00, 4e+2),
        'sl1_fcy': (0.00, 4e+2),
        'sl1_bt0': (0.00, 4.00),

        'urr_dth': (0.00, 2e+1),
        'lrr_dth': (0.00, 5.00),

        'urr_rsr': (0.00, 1.00),
        'urr_tdh': (0.00, 1e+2),
        'urr_tdr': (0.00, 1.00),
        'urr_cst': (0.00, 1.00),
        'urr_dro': (0.00, 1.00),
        'urr_ulc': (0.00, 1.00),

        'lrr_tdh': (0.00, 1e+4),
        'lrr_cst': (0.00, 1.00),
        'lrr_dro': (0.00, 1.00),
    }
bounds_dict_soil0 = {
        'snw_dth': (0.00, 0.00),
        'snw_ast': (-1.0, +1.0),
        'snw_amt': (-0.0, +2.0),
        'snw_amf': (0.00, 2.00),
        'snw_pmf': (0.00, 2.00),

        'sl0_mse': (0, 0), # wie tief der Boden Wasser speichern kann, 0 heißt was zu einer schnellen Sättigung und einer schnellen Verdunstung führt.
        'sl1_mse': (0.00, 2e+2),

        'sl0_fcy': (0.00, 0.00), # Boden weniger Wasser halten kann, was zu schnellerem Abfluss und geringerer Wasserretention führt.
        # Dies fördert eher die Verdunstung als die Speicherung
        'sl0_bt0': (0.00, 0.00), # wie schnell der Boden Wasser aufnimmt
        # niedriger Wert für dazu uf, was zu schnellerem Abfluss und weniger Verdunstung führen kann.

        'sl1_pwp': (0.00, 4e+2),
        'sl1_fcy': (0.00, 4e+2),
        'sl1_bt0': (0.00, 4.00),

        'urr_dth': (0.00, 2e+1),
        'lrr_dth': (0.00, 5.00),

        'urr_rsr': (0.00, 1.00),
        'urr_tdh': (0.00, 1e+2),
        'urr_tdr': (0.00, 1.00),
        'urr_cst': (0.00, 1.00),
        'urr_dro': (0.00, 1.00),
        'urr_ulc': (0.00, 1.00),

        'lrr_tdh': (0.00, 1e+4),
        'lrr_cst': (0.00, 1.00),
        'lrr_dro': (0.00, 1.00),
    }

bounds_dict_soil1 = {
        'snw_dth': (0.00, 0.00),
        'snw_ast': (-1.0, +1.0),
        'snw_amt': (-0.0, +2.0),
        'snw_amf': (0.00, 2.00),
        'snw_pmf': (0.00, 2.00),

        'sl0_mse': (0.00, 1e+2),
        'sl1_mse': (0.00, 0.00), # wie tief der Boden Wasser speichern kann, schnellen Versickerung und einem schnelleren Abfluss

        'sl0_fcy': (0.00, 2e+2),
        'sl0_bt0': (0.00, 3.00),

        'sl1_pwp': (0, 0), # begrenzt Pflanzenverfügbarkeit von Wasser und verringert Speicherung
        'sl1_fcy': (0, 0), # wie viel Wasser Boden halten kann, kleiner Wert heißt, weniger Wasser wird gespeichert.
        'sl1_bt0': (0, 0), # Rate, wie viel Wasser in die tiefer Schicht eindringen kann

        'urr_dth': (0.00, 2e+1),
        'lrr_dth': (0.00, 5.00),

        'urr_rsr': (0.00, 1.00),
        'urr_tdh': (0.00, 1e+2),
        'urr_tdr': (0.00, 1.00),
        'urr_cst': (0.00, 1.00),
        'urr_dro': (0.00, 1.00),
        'urr_ulc': (0.00, 1.00),

        'lrr_tdh': (0.00, 1e+4),
        'lrr_cst': (0.00, 1.00),
        'lrr_dro': (0.00, 1.00),
    }

bounds_dict_urr = {
        'snw_dth': (0.00, 0.00),
        'snw_ast': (-1.0, +1.0),
        'snw_amt': (-0.0, +2.0),
        'snw_amf': (0.00, 2.00),
        'snw_pmf': (0.00, 2.00),

        'sl0_mse': (0.00, 1e+2),
        'sl1_mse': (0.00, 2e+2),

        'sl0_fcy': (0.00, 2e+2),
        'sl0_bt0': (0.00, 3.00),

        'sl1_pwp': (0.00, 4e+2),
        'sl1_fcy': (0.00, 4e+2),
        'sl1_bt0': (0.00, 4.00),

        'urr_dth': (0, 0), #initial water level in upper reservoir
        'lrr_dth': (0.00, 5.00),

        'urr_rsr': (0, 0), # Runoff Split Factor, höherer Wert bedeutet mehr schneller Abfluss
        'urr_tdh': (0, 0), # Beginn des schnellen Abflusses
        'urr_tdr': (0, 0), # höherer Wert dieser Konstante bewirkt, dass das Wasser länger im Reservoir verbleibt, bevor schneller Abfluss
        'urr_cst': (1, 1), # Faktor für die Geschwindigkeit
        'urr_dro': (1, 1), # Verhältnis Wasservolumen, dass abfliest
        'urr_ulc': (0, 0), # es fließt dadurch nichts ins untere reservoir, sondern alles in den abfluss

        'lrr_tdh': (0.00, 1e+4),
        'lrr_cst': (0.00, 1.00),
        'lrr_dro': (0.00, 1.00),
    }

bounds_dict_lrr = {
    'snw_dth': (0.00, 0.00),
    'snw_ast': (-1.0, +1.0),
    'snw_amt': (-0.0, +2.0),
    'snw_amf': (0.00, 2.00),
    'snw_pmf': (0.00, 2.00),

    'sl0_mse': (0.00, 1e+2),
    'sl1_mse': (0.00, 2e+2),

    'sl0_fcy': (0.00, 2e+2),
    'sl0_bt0': (0.00, 3.00),

    'sl1_pwp': (0.00, 4e+2),
    'sl1_fcy': (0.00, 4e+2),
    'sl1_bt0': (0.00, 4.00),

    'urr_dth': (0.00, 2e+1),
    'lrr_dth': (0, 0),

    'urr_rsr': (0.00, 1.00),
    'urr_tdh': (0.00, 1e+2),
    'urr_tdr': (0.00, 1.00),
    'urr_cst': (0.00, 1.00),
    'urr_dro': (0.00, 1.00),
    'urr_ulc': (0.00, 1.00),

    'lrr_tdh': (0, 0), #Schwellentiefe bestimmt, ab welchem Füllstand des Reservoirs Wasser abfließen kann
    'lrr_cst': (1, 1), # wie schnell oder langsam, sehr hohen Wert: Beschleunigt den Abfluss, sodass er direkt erfolgt.
    'lrr_dro': (1, 1), #Discharge Ratio führt dazu, dass ein größerer Anteil des Wassers aus dem Reservoir abgeführt wird,
}