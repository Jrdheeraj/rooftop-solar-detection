import React, { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { FiArrowUpRight } from 'react-icons/fi';

const IMG_PADDING = 12;

const STEPS = [
  {
    imgUrl: 'https://images.unsplash.com/photo-1509395176047-4a66953fd231?auto=format&fit=crop&w=1600&q=80',
    subheading: 'Step 1',
    heading: 'Upload or Select Location',
    title: 'Input Your Data',
    desc: 'Upload a rooftop image or enter latitude and longitude coordinates. Our system fetches high-resolution satellite imagery and prepares it for analysis.',
  },
  {
    imgUrl: '/ai-network.jpg',
    subheading: 'Step 2',
    heading: 'AI Detects Solar Panels',
    title: 'AI Processing',
    desc: 'Our YOLO-based AI model scans the image, identifies solar panels, and calculates their position, overlap, and confidence score.',
  },
  {
    imgUrl: 'https://images.unsplash.com/photo-1508514177221-188b1cf16e9d?auto=format&fit=crop&w=1600&q=80',
    subheading: 'Step 3',
    heading: 'Get Smart Insights',
    title: 'Visual Results',
    desc: 'View detected panels, estimated solar area, and performance metrics through a clean visual dashboard with overlay highlights.',
  },
];

/* ─── Main Section ─── */
const HowItWorks = () => {
  return (
    <div style={{ background: '#ffffff' }}>
      {/* Section header */}
      <div style={{
        textAlign: 'center',
        padding: '5rem 1.5rem 2rem',
      }}>
        <p style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: '1.15rem',
          fontWeight: 600,
          letterSpacing: '0.15em',
          textTransform: 'uppercase',
          color: 'hsl(152 60% 32%)',
          marginBottom: '0.75rem',
        }}>
          How It Works
        </p>
        <h2 style={{
          fontFamily: "'Inter', sans-serif",
          fontSize: 'clamp(2rem, 4vw, 3.2rem)',
          fontWeight: 800,
          letterSpacing: '-0.03em',
          color: '#141414',
          lineHeight: 1.1,
        }}>
          Three steps to solar intelligence
        </h2>
      </div>

      {/* Parallax cards */}
      {STEPS.map((step, i) => (
        <TextParallaxContent
          key={i}
          imgUrl={step.imgUrl}
          subheading={step.subheading}
          heading={step.heading}
        >
          <StepContent title={step.title} desc={step.desc} isLast={i === STEPS.length - 1} />
        </TextParallaxContent>
      ))}
    </div>
  );
};

/* ─── Parallax Image + Sticky Text ─── */
const TextParallaxContent = ({ imgUrl, subheading, heading, children }) => {
  return (
    <div style={{ paddingLeft: IMG_PADDING, paddingRight: IMG_PADDING }}>
      <div style={{ position: 'relative', height: '150vh' }}>
        <StickyImage imgUrl={imgUrl} />
        <OverlayContent heading={heading} subheading={subheading} />
      </div>
      {children}
    </div>
  );
};

/* ─── Sticky + Parallax Image ─── */
const StickyImage = ({ imgUrl }) => {
  const targetRef = useRef(null);
  
  const { scrollYProgress: exitProgress } = useScroll({
    target: targetRef,
    offset: ['end end', 'end start'],
  });

  const { scrollYProgress: entryProgress } = useScroll({
    target: targetRef,
    offset: ['start end', 'start start'],
  });

  const mainOpacity = useTransform(entryProgress, [0, 0.4], [0, 1]);
  const scaleFromEntry = useTransform(entryProgress, [0, 1], [0.05, 1]);
  const scaleToExit = useTransform(exitProgress, [0, 1], [1, 0.85]);
  
  // Consolidate scale to avoid custom function callback that recalculates every frame
  const scale = useTransform(
    [entryProgress, exitProgress],
    ([entry, exit]) => {
      if (exit > 0) return 1 - (exit * 0.15);
      return 0.05 + (entry * 0.95);
    }
  );

  const overlayOpacity = useTransform(exitProgress, [0, 1], [1, 0]);

  return (
    <motion.div
      style={{
        backgroundImage: `url(${imgUrl})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        height: `calc(100vh - ${IMG_PADDING * 2}px)`,
        top: IMG_PADDING,
        position: 'sticky',
        borderRadius: 16,
        overflow: 'hidden',
        transformOrigin: 'center center',
        scale,
        opacity: mainOpacity,
        willChange: 'transform, opacity',
      }}
      ref={targetRef}
    >
      <motion.div
        style={{
          position: 'absolute',
          inset: 0,
          background: 'linear-gradient(to bottom, rgba(0,0,0,0.35), rgba(0,0,0,0.55))',
          opacity: overlayOpacity,
        }}
      />
    </motion.div>
  );
};

/* ─── Overlay Heading on Image ─── */
const OverlayContent = ({ subheading, heading }) => {
  const targetRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: targetRef,
    offset: ['start end', 'end start'],
  });

  const y = useTransform(scrollYProgress, [0, 1], [250, -250]);
  const opacityHeading = useTransform(scrollYProgress, [0.25, 0.5, 0.75], [0, 1, 0]);

  return (
    <motion.div
      ref={targetRef}
      style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 10,
        willChange: 'transform, opacity',
      }}
    >
      <motion.p
        style={{
          y,
          opacity: opacityHeading,
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: '0.85rem',
          fontWeight: 500,
          letterSpacing: '0.12em',
          textTransform: 'uppercase',
          color: 'hsl(152 60% 55%)',
          marginBottom: '0.75rem',
          textAlign: 'center',
        }}
      >
        {subheading}
      </motion.p>
      <motion.p
        style={{
          y,
          opacity: opacityHeading,
          fontFamily: "'Inter', sans-serif",
          fontSize: 'clamp(2rem, 5vw, 4rem)',
          fontWeight: 800,
          letterSpacing: '-0.03em',
          color: 'white',
          textAlign: 'center',
          maxWidth: '700px',
          lineHeight: 1.05,
          padding: '0 1rem',
        }}
      >
        {heading}
      </motion.p>
    </motion.div>
  );
};

/* ─── Step Content Below Each Card ─── */
const StepContent = ({ title, desc, isLast }) => (
  <div style={{
    maxWidth: '72rem',
    margin: '0 auto',
    display: 'grid',
    gridTemplateColumns: '1fr',
    gap: '2rem',
    padding: '3rem 1.5rem 6rem',
  }}>
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
      gap: '2rem',
      alignItems: 'start',
    }}>
      <h3 style={{
        fontFamily: "'Inter', sans-serif",
        fontSize: 'clamp(1.5rem, 3vw, 2rem)',
        fontWeight: 700,
        color: '#141414',
        letterSpacing: '-0.02em',
        lineHeight: 1.2,
      }}>
        {title}
      </h3>
      <div>
        <p style={{
          fontFamily: "'Inter', sans-serif",
          fontSize: 'clamp(1rem, 1.5vw, 1.2rem)',
          color: '#555',
          lineHeight: 1.7,
          marginBottom: '1.5rem',
        }}>
          {desc}
        </p>
        <button style={{
          fontFamily: "'Inter', sans-serif",
          display: 'inline-flex',
          alignItems: 'center',
          gap: '6px',
          background: 'hsl(152 60% 32%)',
          color: 'white',
          border: 'none',
          borderRadius: '9999px',
          padding: '12px 28px',
          fontSize: '0.95rem',
          fontWeight: 600,
          cursor: 'pointer',
          letterSpacing: '-0.01em',
          transition: 'background 0.2s ease',
        }}
          onMouseEnter={(e) => e.target.style.background = 'hsl(152 60% 26%)'}
          onMouseLeave={(e) => e.target.style.background = 'hsl(152 60% 32%)'}
        >
          {isLast ? 'Get Started' : 'Learn More'} <FiArrowUpRight />
        </button>
      </div>
    </div>
  </div>
);

export default HowItWorks;
