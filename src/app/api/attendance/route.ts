import { NextResponse } from 'next/server';
import { prisma } from '../../../lib/prisma';

function normalizeVietnameseName(name: string): string {
  // Basic text normalization - remove extra spaces and convert to lowercase
  return name.toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();
}

function calculateNameSimilarity(name1: string, name2: string): number {
  const norm1 = normalizeVietnameseName(name1);
  const norm2 = normalizeVietnameseName(name2);

  // Split into words and handle different word orders using arrays
  const words1 = norm1.split(' ');
  const words2 = norm2.split(' ');

  // Calculate similarity using array operations
  const intersection = words1.filter(x => words2.includes(x));
  const union = Array.from(new Set([...words1, ...words2]));

  return intersection.length / union.length;
}

export async function POST(request: Request) {
  try {
    const uploadedFormData = await request.formData();
    const file = uploadedFormData.get('file') as unknown as File;

    if (!file) {
      return NextResponse.json({ error: 'No file uploaded' }, { status: 400 });
    }

    // Convert the File object to a Buffer/Blob that can be sent to Python
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    // Create a new FormData for the Python service
    const pythonFormData = new FormData();
    pythonFormData.append('file', new Blob([buffer], { type: file.type }), file.name);    // Forward the image to Python Mistral OCR service
    const ocrResponse = await fetch('http://localhost:8000/api/ocr', {
      method: 'POST',
      body: pythonFormData,
    });

    if (!ocrResponse.ok) {
      const errorText = await ocrResponse.text();
      console.error('OCR Service error:', errorText);
      return NextResponse.json({ 
        error: 'OCR Service failed to process image',
        details: errorText
      }, { status: 500 });
    }

    const ocrResult = await ocrResponse.json();
    console.log('OCR Result:', ocrResult);

    // Check OCR confidence threshold
    if (ocrResult.confidence < 50) {  // Minimum 50% confidence required
      return NextResponse.json({ 
        error: 'Low confidence OCR result',
        ocrResult: {
          confidence: ocrResult.confidence,
          ocrText: ocrResult.ocr_text
        }
      }, { status: 400 });
    }

    // Try to find student by ID or name
    let student = null;
    let matchType = '';

    // First try to find by student ID
    if (ocrResult.student_id) {
      student = await prisma.student.findFirst({
        where: { studentId: ocrResult.student_id }
      });
      if (student) matchType = 'ID_Match';
    }

    // If no match by ID and OCR confidence is high enough for name matching
    if (!student && ocrResult.student_name && ocrResult.confidence >= 75) {  // Higher threshold for name matching
      const allStudents = await prisma.student.findMany();
      
      let bestMatch = null;
      let bestSimilarity = 0;
      
      for (const potentialStudent of allStudents) {
        const similarity = calculateNameSimilarity(potentialStudent.name, ocrResult.student_name);
        if (similarity > bestSimilarity && similarity >= 0.7) {
          bestSimilarity = similarity;
          bestMatch = potentialStudent;
        }
      }
      
      if (bestMatch) {
        student = bestMatch;
        matchType = 'Name_Match';
      }
    }

    // If no student found by either method
    if (!student) {
      return NextResponse.json({ 
        error: 'Student not found',
        ocrResult: {
          studentId: ocrResult.student_id,
          studentName: ocrResult.student_name,
          confidence: ocrResult.confidence,
          ocrText: ocrResult.ocr_text
        }
      }, { status: 404 });
    }

    // Create attendance record
    const attendance = await prisma.attendance.create({
      data: {
        studentId: student.id,
        status: matchType,
        date: new Date()
      }
    });

    return NextResponse.json({ 
      success: true, 
      attendance,
      matchType,
      student: {
        id: student.id,
        studentId: student.studentId,
        name: student.name,
        class: student.class
      },
      ocrResult: {
        studentId: ocrResult.student_id,
        studentName: ocrResult.student_name,
        confidence: ocrResult.confidence,
        ocrText: ocrResult.ocr_text
      }
    });
  } catch (error) {
    console.error('Error processing attendance:', error);
    return NextResponse.json({ 
      error: 'Failed to process attendance',
      errorDetails: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}