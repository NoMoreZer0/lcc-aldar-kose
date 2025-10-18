import { useEffect, useState, useCallback } from 'react';
import type { MessageAttachment } from '../types';

interface CinemaModeProps {
  frames: MessageAttachment[];
  prompt: string;
  isOpen: boolean;
  onClose: () => void;
  initialFrame?: number;
}

export const CinemaMode = ({ frames, prompt, isOpen, onClose, initialFrame = 0 }: CinemaModeProps) => {
  const [currentIndex, setCurrentIndex] = useState(initialFrame);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showMetadata, setShowMetadata] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState<'slow' | 'normal' | 'fast'>('normal');
  const [isTransitioning, setIsTransitioning] = useState(false);

  const imageFrames = frames.filter((f) => f.type === 'image');
  const totalFrames = imageFrames.length;
  const currentFrame = imageFrames[currentIndex];

  const SPEEDS = {
    slow: 5000,
    normal: 3000,
    fast: 1500,
  };

  const goToFrame = useCallback((index: number) => {
    if (index < 0 || index >= totalFrames) return;
    setIsTransitioning(true);
    setTimeout(() => {
      setCurrentIndex(index);
      setIsTransitioning(false);
    }, 300);
  }, [totalFrames]);

  const nextFrame = useCallback(() => {
    const next = (currentIndex + 1) % totalFrames;
    goToFrame(next);
  }, [currentIndex, totalFrames, goToFrame]);

  const prevFrame = useCallback(() => {
    const prev = currentIndex === 0 ? totalFrames - 1 : currentIndex - 1;
    goToFrame(prev);
  }, [currentIndex, totalFrames, goToFrame]);

  const togglePlayback = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  const downloadFrame = useCallback(async () => {
    if (!currentFrame) return;

    try {
      const response = await fetch(currentFrame.url);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `frame-${currentIndex + 1}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to download frame:', error);
    }
  }, [currentFrame, currentIndex]);

  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(console.error);
    } else {
      document.exitFullscreen().catch(console.error);
    }
  }, []);

  // Auto-play effect
  useEffect(() => {
    if (!isPlaying || !isOpen) return;

    const interval = setInterval(() => {
      nextFrame();
    }, SPEEDS[playbackSpeed]);

    return () => clearInterval(interval);
  }, [isPlaying, isOpen, nextFrame, playbackSpeed]);

  // Keyboard shortcuts
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'Escape':
          onClose();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          prevFrame();
          break;
        case 'ArrowRight':
          e.preventDefault();
          nextFrame();
          break;
        case ' ':
          e.preventDefault();
          togglePlayback();
          break;
        case 'f':
        case 'F':
          e.preventDefault();
          toggleFullscreen();
          break;
        case 'i':
        case 'I':
          e.preventDefault();
          setShowMetadata((prev) => !prev);
          break;
        case 'd':
        case 'D':
          e.preventDefault();
          downloadFrame();
          break;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose, prevFrame, nextFrame, togglePlayback, toggleFullscreen, downloadFrame]);

  // Prevent background scroll
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  if (!isOpen || !currentFrame) return null;

  const progress = ((currentIndex + 1) / totalFrames) * 100;

  return (
    <div className="cinema-mode">
      {/* Header */}
      <header className="cinema-mode__header">
        <button
          className="cinema-mode__back-btn"
          onClick={onClose}
          aria-label="Exit cinema mode"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5M12 19l-7-7 7-7" />
          </svg>
          Back
        </button>

        <div className="cinema-mode__title">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
            <rect x="2" y="6" width="4" height="12" rx="1" />
            <rect x="7" y="4" width="4" height="16" rx="1" />
            <rect x="12" y="6" width="4" height="12" rx="1" />
            <rect x="17" y="8" width="4" height="8" rx="1" />
          </svg>
          Cinema Mode
        </div>

        <div className="cinema-mode__actions">
          <button
            className="cinema-mode__action-btn"
            onClick={togglePlayback}
            aria-label={isPlaying ? 'Pause' : 'Play'}
            title={`${isPlaying ? 'Pause' : 'Play'} (Space)`}
          >
            {isPlaying ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="4" width="4" height="16" rx="1" />
                <rect x="14" y="4" width="4" height="16" rx="1" />
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
          </button>

          {isPlaying && (
            <div className="cinema-mode__speed-control">
              {(['slow', 'normal', 'fast'] as const).map((speed) => (
                <button
                  key={speed}
                  className={`cinema-mode__speed-btn ${playbackSpeed === speed ? 'cinema-mode__speed-btn--active' : ''}`}
                  onClick={() => setPlaybackSpeed(speed)}
                >
                  {speed === 'slow' ? '0.5×' : speed === 'normal' ? '1×' : '2×'}
                </button>
              ))}
            </div>
          )}

          <button
            className="cinema-mode__action-btn"
            onClick={downloadFrame}
            aria-label="Download frame"
            title="Download current frame (D)"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" />
            </svg>
          </button>

          <button
            className="cinema-mode__action-btn"
            onClick={toggleFullscreen}
            aria-label="Toggle fullscreen"
            title="Toggle fullscreen (F)"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
            </svg>
          </button>

          <button
            className="cinema-mode__action-btn"
            onClick={() => setShowMetadata((prev) => !prev)}
            aria-label="Toggle metadata"
            title="Toggle info (I)"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
          </button>
        </div>
      </header>

      {/* Main viewport */}
      <div className="cinema-mode__viewport">
        <div className={`cinema-mode__frame ${isTransitioning ? 'cinema-mode__frame--transitioning' : ''}`}>
          <img src={currentFrame.url} alt={currentFrame.alt} />
        </div>

        {/* Navigation arrows */}
        <button
          className="cinema-mode__nav cinema-mode__nav--prev"
          onClick={prevFrame}
          aria-label="Previous frame"
        >
          ‹
        </button>
        <button
          className="cinema-mode__nav cinema-mode__nav--next"
          onClick={nextFrame}
          aria-label="Next frame"
        >
          ›
        </button>

        {/* Metadata overlay */}
        {showMetadata && (
          <div className="cinema-mode__metadata">
            <div className="cinema-mode__frame-info">
              Frame {String(currentIndex + 1).padStart(2, '0')} / {String(totalFrames).padStart(2, '0')}
            </div>
            <div className="cinema-mode__caption">{currentFrame.alt}</div>
          </div>
        )}
      </div>

      {/* Progress bar */}
      <div className="cinema-mode__progress-container">
        <div className="cinema-mode__progress-bar" style={{ width: `${progress}%` }} />
      </div>

      {/* Thumbnail strip */}
      <div className="cinema-mode__thumbnails">
        {imageFrames.map((frame, index) => (
          <button
            key={`${frame.url}-${index}`}
            className={`cinema-mode__thumbnail ${index === currentIndex ? 'cinema-mode__thumbnail--active' : ''}`}
            onClick={() => goToFrame(index)}
            aria-label={`Go to frame ${index + 1}`}
          >
            <img src={frame.url} alt="" loading="lazy" />
            <span className="cinema-mode__thumbnail-number">{index + 1}</span>
          </button>
        ))}
      </div>

      {/* Keyboard shortcuts hint */}
      <div className={`cinema-mode__shortcuts ${showMetadata ? 'cinema-mode__shortcuts--visible' : ''}`}>
        <kbd>←</kbd>
        <kbd>→</kbd>
        <span>Navigate</span>
        <span>•</span>
        <kbd>Space</kbd>
        <span>Play/Pause</span>
        <span>•</span>
        <kbd>F</kbd>
        <span>Fullscreen</span>
        <span>•</span>
        <kbd>D</kbd>
        <span>Download</span>
        <span>•</span>
        <kbd>I</kbd>
        <span>Info</span>
        <span>•</span>
        <kbd>ESC</kbd>
        <span>Exit</span>
      </div>

      {/* Prompt display */}
      {prompt && (
        <div className={`cinema-mode__prompt ${showMetadata ? 'cinema-mode__prompt--visible' : ''}`}>
          <div className="cinema-mode__prompt-label">Prompt</div>
          <div className="cinema-mode__prompt-text">{prompt}</div>
        </div>
      )}
    </div>
  );
};