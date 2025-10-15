from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import io

def generate_pdf_report(results, report_title="Crack Detection Report"):
    """
    Generate a PDF report from prediction results.
    
    Args:
        results: List of prediction result dictionaries
        report_title: Title for the PDF report
    
    Returns:
        BytesIO object containing the PDF
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#333333'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    title = Paragraph(report_title, title_style)
    story.append(title)
    
    timestamp = Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles['Normal']
    )
    story.append(timestamp)
    story.append(Spacer(1, 0.3*inch))
    
    summary_heading = Paragraph("Executive Summary", heading_style)
    story.append(summary_heading)
    
    total_images = len(results)
    crack_images = len([r for r in results if r['predicted_class'] != 'No Crack'])
    no_crack_images = total_images - crack_images
    avg_confidence = sum(r['confidence'] for r in results) / total_images if total_images > 0 else 0
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Images Analyzed', str(total_images)],
        ['Images with Cracks', f"{crack_images} ({crack_images/total_images*100:.1f}%)"],
        ['Images without Cracks', f"{no_crack_images} ({no_crack_images/total_images*100:.1f}%)"],
        ['Average Confidence', f"{avg_confidence:.1%}"]
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    
    crack_types = {}
    for r in results:
        crack_type = r['predicted_class']
        crack_types[crack_type] = crack_types.get(crack_type, 0) + 1
    
    if crack_types:
        dist_heading = Paragraph("Crack Type Distribution", heading_style)
        story.append(dist_heading)
        
        dist_data = [['Crack Type', 'Count', 'Percentage']]
        for crack_type, count in sorted(crack_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_images) * 100
            dist_data.append([crack_type, str(count), f"{percentage:.1f}%"])
        
        dist_table = Table(dist_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        dist_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        story.append(dist_table)
        story.append(Spacer(1, 0.3*inch))
    
    results_heading = Paragraph("Detailed Results", heading_style)
    story.append(results_heading)
    
    class_names = ['Longitudinal Crack', 'No Crack', 'Oblique Crack', 'Transverse Crack']
    
    results_data = [['Image', 'Predicted Class', 'Confidence', 'Timestamp']]
    
    for result in results:
        results_data.append([
            result['image_name'][:30],
            result['predicted_class'],
            f"{result['confidence']:.1%}",
            result['timestamp'].strftime('%Y-%m-%d %H:%M')
        ])
    
    results_table = Table(results_data, colWidths=[2*inch, 1.8*inch, 1.2*inch, 1.5*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 0.2*inch))
    
    footer = Paragraph(
        "This report was automatically generated by the Geopolymer Concrete Crack Detection System",
        styles['Normal']
    )
    story.append(footer)
    
    doc.build(story)
    
    buffer.seek(0)
    return buffer
