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
          <span className="cinematic-carousel__alt">{activeFrame.alt}</span>
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
