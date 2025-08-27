def generate_medical_report(results, patient_id):
    report = f"Abdominal CT Scan Radiology Report\n\n"
    report += "Examination Type: Multidetector CT Scan of the Abdomen (Non-Contrast/Contrast-Enhanced)\n\n"
    report += "Clinical Information/Indication:\n"
    report += "CT scan of the abdomen performed to evaluate for potential anomalies in abdominal organs, including the pancreas, spleen, kidneys, gallbladder, liver, and stomach. No specific clinical symptoms reported; routine screening or follow-up evaluation.\n\n"
    report += "Technique:\n"
    report += "Multiphase CT imaging of the abdomen was performed using a multidetector CT scanner. Axial images were obtained from the diaphragm to the pubic symphysis with 5 mm slice thickness. Images were reconstructed in axial, coronal, and sagittal planes. No immediate complications occurred during the procedure.\n\n"
    report += "Findings:\n"

    # فقط ارگان‌های موجود توی results رو پردازش کن
    processed_organs = set()
    for organ in results.keys():
        if organ == "kidneys":
            report += "- **Kidneys:** "
            right_kidney_normal = "Right kidney: Normal. " if "kidney_right" not in results or not results["kidney_right"]["has_anomaly"] else ""
            left_kidney_normal = "Left kidney: Normal. " if "kidney_left" not in results or not results["kidney_left"]["has_anomaly"] else ""
            if right_kidney_normal or left_kidney_normal:
                report += "Bilateral kidneys demonstrate normal size, position, and enhancement. No hydronephrosis, stones, or masses identified. " + right_kidney_normal + left_kidney_normal
            elif "kidney_right" in results and results["kidney_right"]["has_anomaly"]:
                report += f"Abnormalities noted in the right kidney with suspicious findings on slices {', '.join(results['kidney_right']['suspicious_slices'])}. Further characterization is recommended."
            elif "kidney_left" in results and results["kidney_left"]["has_anomaly"]:
                report += f"Abnormalities noted in the left kidney with suspicious findings on slices {', '.join(results['kidney_left']['suspicious_slices'])}. Further characterization is recommended."
            report += "\n"
            processed_organs.add("kidneys")
        else:
            if organ in results:
                report += f"- **{organ.capitalize()}:** "
                if results[organ]["has_anomaly"]:
                    slices = results[organ]["suspicious_slices"]
                    slice_ranges = []
                    start = slices[0]
                    for i in range(1, len(slices)):
                        if int(slices[i]) != int(slices[i-1]) + 1:
                            if start == slices[i-1]:
                                slice_ranges.append(start)
                            else:
                                slice_ranges.append(f"{start}-{slices[i-1]}")
                            start = slices[i]
                    if start == slices[-1]:
                        slice_ranges.append(start)
                    else:
                        slice_ranges.append(f"{start}-{slices[-1]}")
                    slice_text = ", ".join(slice_ranges)
                    report += f"Abnormalities noted with suspicious findings suggestive of anomaly. Specific suspicious slices include: {slice_text}. The {organ} parenchyma appears heterogeneous in these regions; further characterization is recommended. No evidence of duct dilatation or calcifications.\n"
                else:
                    report += "Normal appearance with no abnormalities detected.\n"
                processed_organs.add(organ)

    report += "\nImpression:\n"
    impressions = []
    for organ in results.keys():
        if organ in processed_organs:
            if results[organ]["has_anomaly"]:
                slices = results[organ]["suspicious_slices"]
                slice_ranges = []
                start = slices[0]
                for i in range(1, len(slices)):
                    if int(slices[i]) != int(slices[i-1]) + 1:
                        if start == slices[i-1]:
                            slice_ranges.append(start)
                        else:
                            slice_ranges.append(f"{start}-{slices[i-1]}")
                        start = slices[i]
                if start == slices[-1]:
                    slice_ranges.append(start)
                else:
                    slice_ranges.append(f"{start}-{slices[-1]}")
                slice_text = ", ".join(slice_ranges)
                impressions.append(f"Anomaly detected in the {organ} with suspicious findings on multiple slices ({slice_text}), warranting further evaluation with MRI, endoscopy, or biopsy to rule out neoplastic or inflammatory processes.")
            else:
                impressions.append(f"No abnormalities identified in the {organ}.")
    report += "\n".join([f"{i+1}. {imp}" for i, imp in enumerate(impressions)])
    report += "\n\nNote: This report is based on automated anomaly detection from CT imaging. All findings should be correlated with clinical history and confirmed by a qualified physician. If additional details or follow-up are required, please contact the radiology department."

    return report