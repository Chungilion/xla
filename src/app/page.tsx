'use client';

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import Webcam from 'react-webcam';
import { debounce } from 'lodash';

// Types
interface Student {
  id: number;
  studentId: string;
  name: string;
  class: string;
  attendance: AttendanceRecord[];
}

interface AttendanceRecord {
  id: number;
  date: string;
  status: string;
}

interface ProcessedData {
  success?: boolean;
  error?: string;
  confidence: number;
  raw_text: string;
  full_name: string;
  student_id: string;
  birth_date: string;
  class_name: string;
  major: string;
}

// Utilities
const textProcessors = {
  preprocessText: (text: string): string => {
    const corrections: Record<string, string> = {
      '8': 'B', '0': 'D', 'о': 'o', 'О': 'O', 'З': 'E',
      'l': '1', 'I': '1', 'і': 'i', 'А': 'A', 'Е': 'E',
      'T': 'T', 'С': 'C', 'Ѕ': 'S', '5': 'S', '6': 'G'
    };
    return text.split('').map(char => corrections[char] || char).join('');
  },

  cleanOCRText: (text: string): string => {
    if (!text) return '';
    const processed = textProcessors.preprocessText(text);
    return processed
      .split('\n')
      .map(line => line.trim())
      .filter(line => {
        const validChars = /[a-zA-Z0-9\u00C0-\u1EF9]/;
        const minLength = 2;
        return line.length >= minLength && validChars.test(line);
      })
      .join('\n');
  },

  formatOCRSection: (text: string, label: string): string => {
    const matches = text.match(new RegExp(`${label}[:\\s]*(.*?)(?=\\n|$)`, 'i'));
    return matches ? matches[1].trim() : '';
  }
};

const validators = {
  studentId: (id: string): boolean => /^[BDCL]\d{8,10}$/.test(id.toUpperCase()),
  className: (className: string): boolean => 
    /^[EDCL]\d{2}[A-Z]+\d{2,3}(?:-[A-Z])?$/i.test(className.toUpperCase())
};

// Components
const CameraControls = ({ 
  isCameraActive, 
  setIsCameraActive, 
  isAutoDetecting, 
  setIsAutoDetecting,
  zoom,
  setZoom,
  onFileSelect
}: {
  isCameraActive: boolean;
  setIsCameraActive: (active: boolean) => void;
  isAutoDetecting: boolean;
  setIsAutoDetecting: (detecting: boolean) => void;
  zoom: number;
  setZoom: (zoom: number) => void;
  onFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="flex flex-wrap gap-3">
      <button
        onClick={() => setIsCameraActive(!isCameraActive)}
        className="btn-primary px-5 py-2.5 rounded-lg font-medium"
      >
        {isCameraActive ? 'Tắt Camera' : 'Bật Camera'}
      </button>

      <input
        type="file"
        accept="image/*"
        onChange={onFileSelect}
        ref={fileInputRef}
        className="hidden"
      />
      <button
        onClick={() => fileInputRef.current?.click()}
        className="btn-primary px-5 py-2.5 rounded-lg font-medium"
      >
        Tải Ảnh Lên
      </button>

      {isCameraActive && (
        <>
          <button
            onClick={() => setIsAutoDetecting(!isAutoDetecting)}
            className={`px-5 py-2.5 rounded-lg font-medium transition-all ${
              isAutoDetecting 
                ? 'btn-success' 
                : 'bg-gray-500 hover:bg-gray-600 text-white'
            }`}
          >
            {isAutoDetecting ? 'Tự Động Đang Bật' : 'Tự Động Đang Tắt'}
          </button>
          <div className="flex items-center gap-3 ml-auto">
            <label className="text-sm font-medium text-[--text-secondary]">Thu phóng:</label>
            <input
              type="range"
              min="1"
              max="2"
              step="0.1"
              value={zoom}
              onChange={(e) => setZoom(parseFloat(e.target.value))}
              className="w-32 accent-indigo-600"
            />
          </div>
        </>
      )}
    </div>
  );
};

const OCRResults = ({ ocrText, processedData }: { ocrText: string; processedData: ProcessedData | null }) => (
  <div className="flex-1 space-y-4">
    <div className="card p-6 h-full bg-white rounded-lg shadow-sm border border-gray-100">
      <h3 className="text-lg font-semibold mb-4 text-gray-800 flex items-center gap-3">
        <span>Kết Quả OCR</span>
      </h3>
      
      {ocrText && (
        <div className="mt-4">
          <div className="text-sm font-medium text-gray-600 mb-2">Chi tiết văn bản:</div>
          <pre className="whitespace-pre-wrap text-sm bg-gray-50 p-4 rounded-lg font-mono text-gray-700 max-h-[200px] overflow-y-auto">
            {ocrText}
          </pre>
        </div>
      )}

      {!ocrText && (
        <div className="text-gray-500 text-sm italic">
          Chưa có kết quả OCR. Vui lòng tải lên hoặc chụp ảnh thẻ sinh viên.
        </div>
      )}
    </div>
  </div>
);

const StudentList = ({ students }: { students: Student[] }) => (
  <div className="overflow-x-auto rounded-lg border border-gray-200">
    <table className="min-w-full divide-y divide-gray-200">
      <thead>
        <tr>
          <th className="table-header">Mã SV</th>
          <th className="table-header">Họ và Tên</th>
          <th className="table-header">Lớp</th>
          <th className="table-header">Điểm Danh Gần Nhất</th>
          <th className="table-header">Trạng Thái</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-gray-200 bg-white">
        {students.map((student) => {
          const lastAttendance = student.attendance?.[0];
          return (
            <tr key={student.id} className="hover:bg-gray-50 transition-colors">
              <td className="table-cell font-medium text-indigo-600">{student.studentId}</td>
              <td className="table-cell">{student.name}</td>
              <td className="table-cell">{student.class}</td>
              <td className="table-cell">
                {lastAttendance
                  ? new Date(lastAttendance.date).toLocaleDateString('vi-VN')
                  : 'Chưa có'}
              </td>
              <td className="table-cell">
                {lastAttendance?.status === 'ID_Match' ? (
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800">
                    Có mặt (Mã SV)
                  </span>
                ) : lastAttendance?.status === 'Name_Match' ? (
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                    Có mặt (Tên)
                  </span>
                ) : (
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                    Vắng mặt
                  </span>
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  </div>
);

// Main component
export default function Home() {
  // State
  const [students, setStudents] = useState<Student[]>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [isAutoDetecting, setIsAutoDetecting] = useState(true);
  const [ocrText, setOcrText] = useState<string>('');
  const [currentProcessedData, setCurrentProcessedData] = useState<ProcessedData | null>(null);

  // Refs
  const webcamRef = useRef<Webcam | null>(null);
  const processingRef = useRef(false);

  // API calls with retry and error handling
  const fetchWithRetry = useCallback(async (url: string, options: RequestInit, retries = 3): Promise<Response> => {
    for (let i = 0; i < retries; i++) {
      try {
        const response = await fetch(url, options);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return response;
      } catch (error) {
        if (i === retries - 1) throw error;
        await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
      }
    }
    throw new Error('Failed after retries');
  }, []);

  // Fetch students
  const fetchStudents = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetchWithRetry('/api/students', { method: 'GET' });
      const data = await response.json();
      setStudents(data);
    } catch (error) {
      console.error('Error fetching students:', error);
      setMessage('Lỗi khi tải danh sách sinh viên');
    } finally {
      setLoading(false);
    }
  }, [fetchWithRetry]);

  // Process image with optimized error handling
  const processImage = useCallback(async (file: File) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      // Direct OCR processing
      const ocrResponse = await fetch('http://localhost:8000/api/ocr', {
        method: 'POST',
        body: formData,
      });

      if (!ocrResponse.ok) {
        const errorText = await ocrResponse.text();
        console.error('OCR Error:', errorText);
        throw new Error('OCR processing failed');
      }

      const ocrResult = await ocrResponse.json();
      console.log('Raw OCR Response:', ocrResult);
      
      // Ensure we have the data we need
      if (!ocrResult) {
        throw new Error('Empty OCR response');
      }
      
      // Log the exact structure and values
      console.log('OCR Result Structure:', {
        success: ocrResult.success,
        confidence: ocrResult.confidence,
        raw_text: ocrResult.raw_text,
        full_name: ocrResult.full_name,
        student_id: ocrResult.student_id,
        birth_date: ocrResult.birth_date,
        class_name: ocrResult.class_name,
        major: ocrResult.major
      });

      // Update OCR results immediately
      const cleanedText = textProcessors.cleanOCRText(ocrResult.raw_text || '');
      setOcrText(cleanedText);
      setCurrentProcessedData({
        success: true,
        confidence: ocrResult.confidence || 0,
        raw_text: cleanedText,
        full_name: ocrResult.full_name || '',
        student_id: ocrResult.student_id || '',
        birth_date: ocrResult.birth_date || '',
        class_name: ocrResult.class_name || '',
        major: ocrResult.major || ''
      });

      // Only attempt attendance if we found a student ID and have high confidence
      if (ocrResult.student_id && ocrResult.confidence >= 50) {
        // Now handle attendance
        const response = await fetchWithRetry('/api/attendance', {
          method: 'POST',
          body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
          const formattedMessage = [
            'Điểm danh thành công!',
            ocrResult.student_id && `Mã SV: ${ocrResult.student_id}`,
            ocrResult.full_name && `Họ tên: ${ocrResult.full_name}`,
            ocrResult.class_name && `Lớp: ${ocrResult.class_name}`
          ].filter(Boolean).join('\n');
          
          setMessage(formattedMessage);
          fetchStudents();
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setMessage('Lỗi khi xử lý ảnh');
    }
  }, [fetchWithRetry, fetchStudents]);

  // Debounced frame processing with immediate feedback
  const processFrame = useMemo(() => 
    debounce(async (imageSrc: string) => {
      if (!isAutoDetecting) return;
      if (processingRef.current) return;

      processingRef.current = true;
      try {
        const blob = await fetch(imageSrc).then(r => r.blob());
        const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
        await processImage(file);
      } catch (error) {
        console.error('Error:', error);
      } finally {
        processingRef.current = false;
      }
    }, 200), // Reduced debounce time for more responsive feedback
    [isAutoDetecting, processImage]
  );

  // Camera frame capture
  useEffect(() => {
    if (!isCameraActive || !webcamRef.current) return;

    const interval = setInterval(() => {
      const imageSrc = webcamRef.current?.getScreenshot();
      if (imageSrc) {
        processFrame(imageSrc);
      }
    }, 300); // Capture frame every 300ms

    return () => clearInterval(interval);
  }, [isCameraActive, processFrame]);

  // Clear OCR results when camera is deactivated
  useEffect(() => {
    if (!isCameraActive) {
      setOcrText('');
      setCurrentProcessedData(null);
      setMessage('');
    }
  }, [isCameraActive]);

  // File upload handler
  const handleFileUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
      setMessage('Lỗi: Vui lòng chọn file ảnh');
      return;
    }

    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      setMessage('Lỗi: Kích thước file không được vượt quá 10MB');
      return;
    }

    await processImage(file);
  }, [processImage]);

  // Add student handler with validation
  const addStudent = useCallback(async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const studentId = formData.get('studentId')?.toString().toUpperCase() || '';
    const className = formData.get('class')?.toString().toUpperCase() || '';
    
    if (!validators.studentId(studentId)) {
      setMessage('Lỗi: Mã sinh viên không đúng định dạng (VD: B21DCVT020)');
      return;
    }

    if (!validators.className(className)) {
      setMessage('Lỗi: Mã lớp không đúng định dạng (VD: E21CQCN02-B)');
      return;
    }
    
    try {
      setLoading(true);
      const studentData = {
        studentId,
        name: formData.get('name')?.toString(),
        class: className
      };

      const response = await fetchWithRetry('/api/students', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(studentData)
      });

      const result = await response.json();

      if (response.ok) {
        fetchStudents();
        (e.target as HTMLFormElement).reset();
        setMessage('Thêm sinh viên thành công!');
      } else {
        setMessage(`Lỗi: ${result.error || 'Không thể thêm sinh viên'}`);
      }
    } catch (error) {
      console.error('Error adding student:', error);
      setMessage('Lỗi khi thêm sinh viên');
    } finally {
      setLoading(false);
    }
  }, [fetchWithRetry, fetchStudents]);

  // Initial data fetch
  useEffect(() => {
    fetchStudents();
  }, [fetchStudents]);

  return (
    <main className="min-h-screen p-6 md:p-8 bg-[--background-color] font-['Be_Vietnam_Pro']">
      <h1 className="text-4xl font-bold mb-8 text-[--text-primary] text-center">
        Hệ Thống Điểm Danh Sinh Viên
      </h1>

      {/* Camera Section */}
      <div className="card mb-8 p-6 relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-indigo-500 to-purple-500"></div>
        <h2 className="text-2xl font-semibold mb-6 text-[--text-primary]">Camera Điểm Danh</h2>
        <div className="flex flex-col lg:flex-row gap-6">
          <div className="flex flex-col gap-4 lg:w-[600px]">
            <CameraControls
              isCameraActive={isCameraActive}
              setIsCameraActive={setIsCameraActive}
              isAutoDetecting={isAutoDetecting}
              setIsAutoDetecting={setIsAutoDetecting}
              zoom={zoom}
              setZoom={setZoom}
              onFileSelect={handleFileUpload}
            />
            
            {isCameraActive && (
              <div className="camera-container rounded-lg overflow-hidden bg-black relative">
                <div 
                  style={{ 
                    transform: `scale(${zoom})`, 
                    transformOrigin: 'center',
                    height: '400px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                >
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    screenshotFormat="image/jpeg"
                    mirrored={false}
                    className="h-full object-contain"
                    videoConstraints={{
                      width: 1280,
                      height: 720,
                      facingMode: "user"
                    }}
                  />
                </div>
                {isAutoDetecting && (
                  <div className="absolute top-4 right-4 flex items-center space-x-2 bg-black/50 text-white px-3 py-1.5 rounded-full text-sm">
                    <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                    <span>Đang quét</span>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="lg:w-[400px] flex-shrink-0">
            <OCRResults
              ocrText={ocrText}
              processedData={currentProcessedData}
            />
          </div>
        </div>
        
        {message && (
          <div className="mt-6 p-4 rounded-lg bg-indigo-50 border-l-4 border-indigo-500">
            <p className="text-sm text-indigo-900 whitespace-pre-line">{message}</p>
          </div>
        )}
      </div>

      {/* Add Student Form */}
      <div className="card mb-8 p-6 relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-emerald-500 to-teal-500"></div>
        <h2 className="text-2xl font-semibold mb-6 text-[--text-primary]">Thêm Sinh Viên Mới</h2>
        <form onSubmit={addStudent} className="flex flex-wrap gap-4">
          <div className="flex-1 min-w-[200px]">
            <label className="block text-sm font-medium mb-2 text-[--text-secondary]">Mã SV</label>
            <input
              type="text"
              name="studentId"
              required
              className="input-field w-full"
              placeholder="Nhập mã sinh viên"
            />
          </div>
          <div className="flex-1 min-w-[200px]">
            <label className="block text-sm font-medium mb-2 text-[--text-secondary]">Họ và Tên</label>
            <input
              type="text"
              name="name"
              required
              className="input-field w-full"
              placeholder="Nhập tên sinh viên"
            />
          </div>
          <div className="flex-1 min-w-[200px]">
            <label className="block text-sm font-medium mb-2 text-[--text-secondary]">Lớp</label>
            <input
              type="text"
              name="class"
              required
              className="input-field w-full"
              placeholder="VD: E21CQCN02-B"
            />
          </div>
          <div className="w-full sm:w-auto flex items-end">
            <button
              type="submit"
              disabled={loading}
              className={`btn-success w-full sm:w-auto px-6 py-2.5 rounded-lg font-medium ${
                loading ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {loading ? 'Đang xử lý...' : 'Thêm Sinh Viên'}
            </button>
          </div>
        </form>
      </div>

      {/* Student List */}
      <div className="card p-6 relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-gray-500 to-gray-600"></div>
        <h2 className="text-2xl font-semibold mb-6 text-[--text-primary]">Danh Sách Sinh Viên</h2>
        {loading ? (
          <div className="text-center py-8">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-indigo-500 border-t-transparent"></div>
            <p className="mt-2 text-[--text-secondary]">Đang tải...</p>
          </div>
        ) : (
          <StudentList students={students} />
        )}
      </div>
    </main>
  );
}
