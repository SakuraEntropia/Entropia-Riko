/** Brand logo that accepts ico / png / svg assets.
 *
 * Prefers `public/brand/logo.ico`, then `logo.png`, then `logo.svg`, so the
 * slot works with whichever format is present without changing code. The
 * caller's className carries the squircle clip + sizing.
 */
import { useState } from "react";

const SRC_CHAIN = ["/brand/logo.ico", "/brand/logo.png", "/brand/logo.svg"];

export function BrandLogo({
  className,
  alt = "Entropia Riko logo",
  title,
}: {
  className?: string;
  alt?: string;
  title?: string;
}) {
  const [i, setI] = useState(0);
  return (
    <img
      className={className}
      src={SRC_CHAIN[i]}
      alt={alt}
      title={title}
      onError={() => setI((n) => Math.min(n + 1, SRC_CHAIN.length - 1))}
    />
  );
}
