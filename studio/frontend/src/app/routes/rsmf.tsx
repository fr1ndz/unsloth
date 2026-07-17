/**
 * RSMF Route — TanStack Router file-based route.
 */
import { createFileRoute } from '@tanstack/react-router';
import RsmfPage from '../../features/rsmf/RsmfPage';

export const Route = createFileRoute('/rsmf')({
  component: RsmfPage,
});
