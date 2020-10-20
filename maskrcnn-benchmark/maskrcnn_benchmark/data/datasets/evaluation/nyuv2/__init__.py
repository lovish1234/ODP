from .nyuv2_eval import do_nyuv2_evaluation


def nyuv2_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    expected_results=None,
    expected_results_sigma_tol=None,
):
    return do_nyuv2_evaluation(
        dataset=dataset,
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
