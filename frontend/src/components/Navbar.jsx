import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiChevronDown, FiArrowRight, FiZap, FiMenu, FiX } from 'react-icons/fi';

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <nav className="absolute top-0 left-0 right-0 z-[100] bg-transparent">
        <div className="max-w-7xl mx-auto px-6 md:px-8 h-20 flex items-center justify-between">
          {/* LOGO */}
          <a href="#home" className="flex items-center gap-2.5 group cursor-pointer">
            <div className="w-9 h-9 bg-black rounded-xl flex items-center justify-center text-white shadow-lg group-hover:scale-110 transition-transform duration-300">
              <FiZap className="text-xl" />
            </div>
            <span className="text-xl font-black tracking-tight text-slate-900 font-inter">
              SolarScan
            </span>
          </a>

          {/* DESKTOP NAV LINKS */}
          <div className="hidden md:flex items-center gap-10">
            <NavLink label="Home" href="#home" />
            <NavLink label="How it Works" href="#how-it-works" />
            <NavLink label="Analyze" href="#analyze" />
            <NavLink label="Features" href="#features" />
            <NavLink label="Solutions" href="#solutions" hasDropdown />
            <NavLink label="Contact" href="#about" />
          </div>

          {/* DESKTOP CTA BUTTON */}
          <motion.a
            href="#analyze"
            whileHover={{ scale: 1.05, backgroundColor: '#333' }}
            whileTap={{ scale: 0.95 }}
            className="hidden md:flex bg-black text-white px-7 py-3 rounded-full font-bold items-center gap-2.5 text-sm transition-all shadow-2xl hover:shadow-black/20"
          >
            Analyze Now
            <FiArrowRight className="text-lg" />
          </motion.a>

          {/* MOBILE MENU BUTTON */}
          <button 
            className="md:hidden text-2xl text-slate-900 p-2"
            onClick={() => setIsOpen(true)}
          >
            <FiMenu />
          </button>
        </div>
      </nav>

      {/* MOBILE MENU CARD (Left Side) */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* BACKDROP */}
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsOpen(false)}
              className="fixed inset-0 bg-black/40 backdrop-blur-sm z-[101] md:hidden"
            />

            {/* SIDE DRAWER */}
            <motion.div
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed top-0 left-0 h-screen w-[85%] max-w-sm bg-white/95 backdrop-blur-2xl z-[102] shadow-2xl flex flex-col md:hidden"
            >
              <div className="px-6 h-20 flex items-center justify-between border-b border-gray-100">
                <a href="#home" className="flex items-center gap-2.5" onClick={() => setIsOpen(false)}>
                  <div className="w-8 h-8 bg-black rounded-lg flex items-center justify-center text-white shadow-md">
                    <FiZap className="text-base" />
                  </div>
                  <span className="text-lg font-black tracking-tight text-slate-900 font-inter">
                    SolarScan
                  </span>
                </a>
                <button 
                  onClick={() => setIsOpen(false)}
                  className="p-2 text-2xl text-slate-500 hover:text-black bg-gray-50 hover:bg-gray-100 rounded-full transition-colors"
                >
                  <FiX />
                </button>
              </div>

              <div className="flex flex-col px-6 py-8 gap-6 overflow-y-auto">
                <MobileNavLink label="Home" href="#home" onClick={() => setIsOpen(false)} />
                <MobileNavLink label="How it Works" href="#how-it-works" onClick={() => setIsOpen(false)} />
                <MobileNavLink label="Analyze" href="#analyze" onClick={() => setIsOpen(false)} />
                <MobileNavLink label="Features" href="#features" onClick={() => setIsOpen(false)} />
                <MobileNavLink label="Solutions" href="#solutions" onClick={() => setIsOpen(false)} />
                <MobileNavLink label="Contact" href="#about" onClick={() => setIsOpen(false)} />
              </div>

              <div className="mt-auto px-6 pb-8">
                <a
                  href="#analyze"
                  onClick={() => setIsOpen(false)}
                  className="w-full bg-black text-white px-6 py-4 rounded-xl font-bold flex justify-center items-center gap-2.5 text-sm shadow-xl"
                >
                  Analyze Now
                  <FiArrowRight className="text-lg" />
                </a>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
};

const NavLink = ({ label, href, hasDropdown }) => (
  <a 
    href={href}
    className="flex items-center gap-1.5 text-slate-500 font-semibold hover:text-black transition-all cursor-pointer group text-sm uppercase tracking-wider"
  >
    <span>{label}</span>
    {hasDropdown && (
      <FiChevronDown className="text-slate-400 group-hover:text-black transition-colors" />
    )}
  </a>
);

const MobileNavLink = ({ label, href, onClick }) => (
  <a 
    href={href}
    onClick={onClick}
    className="text-lg font-bold text-slate-700 hover:text-black transition-colors py-2 border-b border-gray-50 uppercase tracking-widest"
  >
    {label}
  </a>
);

export default Navbar;
