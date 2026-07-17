/**
 * R3F 3D Stratum Visualizer — renders spectral decomposition as interactive 3D scene.
 *
 * Each stratum layer is visualized as a ring of singular values (σ)
 * with U/V basis vectors shown as directional particles.
 * Color encodes coherence health; size encodes spectral energy.
 */
import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Float, Sparkles } from '@react-three/drei';
import * as THREE from 'three';

interface StratumVisualizerProps {
  spectrum: number[][];
  coherenceValues?: number[];
  selectedLayer?: number | null;
}

function StratumRing({
  sigma,
  layerIndex,
  yOffset,
  coherence,
  isSelected,
}: {
  sigma: number[];
  layerIndex: number;
  yOffset: number;
  coherence: number;
  isSelected: boolean;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const count = sigma.length;

  // Color based on coherence: green (healthy) → red (collapsed)
  const color = useMemo(() => {
    const h = coherence * 0.35; // 0→red, 0.35→green
    return new THREE.Color().setHSL(h, 0.9, isSelected ? 0.7 : 0.5);
  }, [coherence, isSelected]);

  // Position instances in a ring
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useFrame((state) => {
    if (!meshRef.current) return;
    const t = state.clock.elapsedTime;

    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2 + t * 0.1;
      const radius = 2 + sigma[i] * 0.5;
      dummy.position.set(
        Math.cos(angle) * radius,
        yOffset + Math.sin(t * 0.5 + i * 0.3) * 0.1,
        Math.sin(angle) * radius,
      );
      const scale = 0.05 + sigma[i] * 0.08;
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    }
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
      <sphereGeometry args={[1, 16, 16]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={isSelected ? 0.8 : 0.3}
        transparent
        opacity={0.9}
      />
    </instancedMesh>
  );
}

function LayerLabel({ index, y }: { index: number; y: number }) {
  return (
    <Text
      position={[3.5, y, 0]}
      fontSize={0.2}
      color="white"
      anchorX="left"
      anchorY="middle"
    >
      {`L${index}`}
    </Text>
  );
}

function Scene({ spectrum, coherenceValues, selectedLayer }: StratumVisualizerProps) {
  const spacing = 1.2;

  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-5, -5, 5]} intensity={0.5} color="#4488ff" />

      {spectrum.map((sigma, i) => {
        const coherence = coherenceValues?.[i] ?? 0.5;
        return (
          <group key={i}>
            <StratumRing
              sigma={sigma}
              layerIndex={i}
              yOffset={(i - spectrum.length / 2) * spacing}
              coherence={coherence}
              isSelected={selectedLayer === i}
            />
            <LayerLabel index={i} y={(i - spectrum.length / 2) * spacing} />
          </group>
        );
      })}

      <Sparkles count={200} scale={8} size={2} speed={0.3} opacity={0.15} color="#6688cc" />
      <OrbitControls enablePan autoRotate autoRotateSpeed={0.5} />
    </>
  );
}

export function StratumVisualizer(props: StratumVisualizerProps) {
  return (
    <div style={{ width: '100%', height: '100%', minHeight: 400 }}>
      <Canvas camera={{ position: [5, 3, 5], fov: 50 }} gl={{ antialias: true, alpha: true }}>
        <Scene {...props} />
      </Canvas>
    </div>
  );
}
