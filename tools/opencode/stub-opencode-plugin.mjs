export function initialize() {}
export async function resolve(specifier, context, nextResolve) {
  if (specifier === "@opencode/plugin") {
    return {
      url: "data:text/javascript," +
        encodeURIComponent("export const Plugin = { define: (x) => x }; export default { Plugin };"),
      shortCircuit: true,
    }
  }
  return nextResolve(specifier, context)
}
