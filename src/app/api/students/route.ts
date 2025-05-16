import { NextResponse } from 'next/server';
import { prisma } from '../../../lib/prisma';

// GET all students with their latest attendance
export async function GET() {
  try {
    const students = await prisma.student.findMany({
      include: {
        attendance: {
          orderBy: {
            date: 'desc'
          },
          take: 1
        }
      }
    });
    return NextResponse.json(students);
  } catch (error) {
    console.error('Error fetching students:', error);
    return NextResponse.json({ error: 'Failed to fetch students' }, { status: 500 });
  }
}

// POST new student
export async function POST(request: Request) {
  try {
    const json = await request.json();
    
    // Validate required fields
    if (!json.studentId || !json.name || !json.class) {
      return NextResponse.json({ 
        error: 'Missing required fields: studentId, name, class' 
      }, { status: 400 });
    }

    const student = await prisma.student.create({
      data: {
        studentId: json.studentId,
        name: json.name,
        class: json.class
      }
    });
    return NextResponse.json(student);
  } catch (error: any) {
    // Handle unique constraint violation
    if (error?.code === 'P2002') {
      return NextResponse.json({ 
        error: 'Student ID already exists' 
      }, { status: 409 });
    }
    console.error('Error creating student:', error);
    return NextResponse.json({ error: 'Failed to create student' }, { status: 500 });
  }
}