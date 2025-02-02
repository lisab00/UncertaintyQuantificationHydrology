# Boundaries for shutting off and some comments
bounds_dict_snw = {
        'snw_dth': (0.00, 0.00),  # set initial snow depth to 0
        'snw_ast': (-9999, -9999),  # set air snow temp very low
        'snw_amt': (-9999, -9999),  # set air melt temp very low
        'snw_amf': (0.00, 0.00),  # set air melt factor to 0
        'snw_pmf': (0.00, 0.00),  # set ppt melt factor to 0

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

        'sl1_pwp': (0.00, 4e+2), # should be zero, as they arenot used if soil 0 is turned off
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

        'urr_dth': (0, 0), # initial water level in upper reservoir
        'lrr_dth': (0.00, 5.00),

        'urr_rsr': (0, 0), # müsste auf 1, damit es in gleicher zeitschritt abfließt, Runoff Split Factor, höherer Wert bedeutet mehr schneller Abfluss am gleichen Tag
        # im Model schauen, ob null oder eins: ist 0 !!
        'urr_tdh': (1E6, 1E6), # maximum depth the reseroivr can hold, 10^6, damit es nie aktiviert wird, damit nicht über threshold kommt und es keinen Zwischenabfluss
        'urr_tdr': (0, 0), # tritt ein, wenn wassermenge über dem threshold (urr_tdh) ist: anteil der menge des wassers, welches abfluss wird (wenn der Eimer voll ist, wie viel überschwappt, abfluss wird
        'urr_cst': (0, 0), # wenn wassermenge unter dem threshold (urr_thd) ist, wie viel von der wassermenge abfluss wird, if 0.8 -> 80% will come out of the reservoir
        'urr_dro': (0, 0), # Verhältnis Wasservolumen, dass abfliest, how much is "surface water" of the water which comes out
        'urr_ulc': (1, 1), # Versickerungsrate, müsste auf 1
    #upper lower constant, i think how much , es fließt dadurch nichts ins untere reservoir, sondern alles in den abfluss

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
    'urr_ulc': (0, 0), # muss hier auch auf null, da man nichts versickert haben möchte

    'lrr_tdh': (0, 0), #maximum depth the reservoir can hold; kann kein wasser reinnehmen,
    'lrr_cst': (0, 0), # auch auf null, da , wie schnell oder langsam, sehr hohen Wert: Beschleunigt den Abfluss, sodass er direkt erfolgt.
    'lrr_dro': (0, 0), #Discharge Ratio führt dazu, dass ein größerer Anteil des Wassers aus dem Reservoir abgeführt wird,
}

