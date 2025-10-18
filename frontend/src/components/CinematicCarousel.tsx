import { useEffect, useMemo, useState } from 'react';
import type { MessageAttachment } from '../types';

interface CinematicCarouselProps {
  frames: MessageAttachment[];
  title?: string;
}

const clampIndex = (index: number, count: number) => {
  if (count === 0) {
    return 0;
  }

  if (index < 0) {
    return count - 1;
  }

  if (index >= count) {
    return 0;
  }

  return index;
};

export const CinematicCarousel = ({ frames, title }: CinematicCarouselProps) => {
  const imageFrames = useMemo(
    () => frames.filter((frame) => frame.type === 'image'),
    [frames]
  );

  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    setCurrentIndex((prev) => clampIndex(prev, imageFrames.length));
  }, [imageFrames.length]);

  if (imageFrames.length === 0) {
    return null;
  }

  const activeFrame = imageFrames[currentIndex];
  const frameLabel = `Frame ${currentIndex + 1} of ${imageFrames.length}`;
  const carouselTitle = title || 'Storyboard frames';

  const handleSelectFrame = (nextIndex: number) => {
    setCurrentIndex(clampIndex(nextIndex, imageFrames.length));
  };

  const handlePrevious = () => {
    setCurrentIndex((prev) => clampIndex(prev - 1, imageFrames.length));
  };

  const handleNext = () => {
    setCurrentIndex((prev) => clampIndex(prev + 1, imageFrames.length));
  };

  const handleDownload = async () => {
    try {
      const response = await fetch(activeFrame.url);
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
      console.error('Failed to download image:', error);
    }
  };

  return (
    <section
      className="cinematic-carousel"
      role="group"
      aria-roledescription="carousel"
      aria-label={carouselTitle}
    >
      <div className="cinematic-carousel__viewport">
        <img src={activeFrame.url} alt={activeFrame.alt} loading="lazy" />
        <div className="cinematic-carousel__overlay">
          <span className="cinematic-carousel__indicator">{frameLabel}</span>
        </div>

        <button
          type="button"
          className="cinematic-carousel__control cinematic-carousel__control--prev"
          onClick={handlePrevious}
          aria-label="Previous frame"
        >
          ‹
        </button>

        <button
          type="button"
          className="cinematic-carousel__control cinematic-carousel__control--next"
          onClick={handleNext}
          aria-label="Next frame"
        >
          ›
        </button>

        <button
          type="button"
          className="cinematic-carousel__download"
          onClick={handleDownload}
          aria-label="Download current frame"
          title="Download image"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" />
          </svg>
        </button>
      </div>

      <div className="cinematic-carousel__thumbnails" aria-hidden="true">
        {imageFrames.map((frame, index) => {
          const isActive = index === currentIndex;
          return (
            <button
              type="button"
              key={`${frame.url}-${index}`}
              className={`cinematic-carousel__thumb${isActive ? ' cinematic-carousel__thumb--active' : ''}`}
              onClick={() => {
                handleSelectFrame(index);
              }}
              tabIndex={-1}
            >
              <img src={frame.url} alt="" loading="lazy" />
            </button>
          );
        })}
      </div>
    </section>
  );
};
