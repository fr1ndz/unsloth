/**
 * RSMF Route — registered as lazy-loaded page component.
 *
 * NOTE: TanStack Router codegen must be re-run after adding this file:
 *   cd frontend && npx @tanstack/router-cli generate
 * Until then, navigation uses string-based routing via window.location.
 */
import RsmfPage from '../../features/rsmf/RsmfPage';

export default function RsmfRoute() {
  return <RsmfPage />;
}
