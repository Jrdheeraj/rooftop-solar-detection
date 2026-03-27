import React from 'react';

const Hero = () => {
  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
      position: 'relative',
    }}>

      {/* Background Video */}
      <video autoPlay muted loop playsInline>
        <source src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260319_015952_e1deeb12-8fb7-4071-a42a-60779fc64ab6.mp4" type="video/mp4" />
      </video>
      <div className="video-overlay"></div>

      {/* HERO CONTENT */}
      <div className="relative z-10 flex-1 flex flex-col items-center justify-center mt-20 lg:mt-0 px-5 lg:px-6 text-center">

        {/* Headline */}
        <h1 className="anim-2" style={{
          fontFamily: "'Inter', sans-serif",
          fontSize: 'clamp(2.5rem, 5.5vw, 5rem)',
          fontWeight: 800,
          lineHeight: 1,
          letterSpacing: '-0.03em',
          color: 'hsl(0 0% 8%)',
          maxWidth: '900px',
          margin: '0 auto',
          textAlign: 'center',
        }}>
          Detect Solar Panels<br />
          from <span style={{ color: 'hsl(152 60% 32%)' }}>Satellite</span>{' '}Imagery
        </h1>

        {/* Subheadline */}
        <p className="anim-3" style={{
          marginTop: '1.25rem',
          fontFamily: "'Inter', sans-serif",
          fontSize: 'clamp(0.95rem, 1.5vw, 1.1rem)',
          fontWeight: 400,
          color: 'hsl(0 0% 45%)',
          maxWidth: '540px',
          lineHeight: 1.7,
          margin: '1.25rem auto 0',
          textAlign: 'center',
        }}>
          Upload any rooftop image or connect aerial/satellite data — our AI instantly maps solar panels, estimates capacity, and generates actionable energy reports.
        </p>

        {/* CTA Buttons */}
        <div className="anim-4" style={{
          marginTop: '1.5rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '14px',
          flexWrap: 'wrap',
        }}>
          <button 
            onClick={() => document.getElementById('analyze')?.scrollIntoView({ behavior: 'smooth' })}
            style={{
              fontFamily: "'Inter', sans-serif",
              color: 'white',
              fontSize: '15px',
              fontWeight: 600,
              padding: '13px 30px',
              borderRadius: '9999px',
              border: 'none',
              cursor: 'pointer',
              background: 'hsl(0 0% 8%)',
              letterSpacing: '-0.01em',
            }}
          >
            → Start Analyzing
          </button> 
        </div>

      </div>
    </div>
  );
};

export default Hero;
