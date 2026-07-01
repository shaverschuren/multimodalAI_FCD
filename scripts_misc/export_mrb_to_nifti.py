"""Export MRB segmentations and volumes to NIfTI using 3D Slicer. (Inside script, see export_mrb_batch.py for batch processing.)"""

import slicer
import sys
import os

mrb_path = sys.argv[1]
out_dir = sys.argv[2]

os.makedirs(out_dir, exist_ok=True)

try:
    # Load MRB
    slicer.util.loadScene(mrb_path)

    # Get reference volume (first scalar volume)
    volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    if not volumes:
        raise RuntimeError("No scalar volumes found in scene.")

    ref_vol = volumes[0]

    # ----------------------
    # Export MRI volumes
    # ----------------------
    for vol in volumes:
        name = vol.GetName().replace(" ", "_")
        out_path = os.path.join(out_dir, f"{name}.nii.gz")
        slicer.util.saveNode(vol, out_path)

    # ----------------------
    # Export segmentations
    # ----------------------
    segmentations = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")

    for seg in segmentations:
        seg_name = seg.GetName().replace(" ", "_")

        labelmap = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode",
            f"{seg_name}_labelmap"
        )

        # Set reference geometry explicitly
        seg.SetReferenceImageGeometryParameterFromVolumeNode(ref_vol)

        # Export all segments
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
            seg, labelmap
        )

        out_path = os.path.join(out_dir, f"{seg_name}.nii.gz")
        slicer.util.saveNode(labelmap, out_path)

    print(f"Exported: {mrb_path}")
    slicer.app.exit()

except Exception as e:
    print(f"Error processing {mrb_path}: {str(e)}")
    slicer.app.exit(1)
