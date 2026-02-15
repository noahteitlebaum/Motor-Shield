import { motion } from 'framer-motion';
import { ReactNode } from 'react';

// 1. Define the types for your props
interface FadeInUpProps {
  children: ReactNode; // ReactNode covers strings, numbers, elements, or arrays
  delay?: number;      // The '?' makes the delay optional
}


export default function FadeInUp({ children, delay = 0 }: FadeInUpProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }} //Start hidden and 20px down
      whileInView={{ opacity: 1, y: 0 }} //Animate to visible and original position
      viewport={{ once: true, amount: 0.2 }} //Only animate once; trigger when 20% visible
      transition={{ 
        duration: 0.6, 
        delay: delay, 
        ease: [0.21, 0.47, 0.32, 0.98] //easing 
      }}
    >
      {children}
    </motion.div>
  );
}